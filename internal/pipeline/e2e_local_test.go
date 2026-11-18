package pipeline

// An end-to-end run of one job, on this machine, through the real pipeline:
// the Buffer TEE's dispatch payload -> materialize -> catalogue entry ->
// evaluator -> dependency scan -> dataset listing, ground-truth matching and
// decryption -> infer.py -> adaptor.py -> evaluate.py -> leaderboard metrics.
// Only the cloud is local: the catalogue is an HTTP server on localhost, and
// the dataset is a plaintext folder that is encrypted on the fly in the
// tanuh-enc-dataset-v1 format (the key "unwrap" just decodes it).
//
// It needs the Python stage runtime, so run it inside the Processing TEE
// image. Skipped unless all of these are set:
//
//	TANUH_E2E_PAYLOAD    dispatch payload JSON, as the Buffer TEE builds it
//	TANUH_E2E_DATASET    the dataset's plaintext folder, as uploaded
//	TANUH_E2E_CATALOGUE  the dataset's cat/item response JSON
//	TANUH_E2E_OUT        (optional) folder to copy results.json into
//
//	go test -c -o pipeline.test ./internal/pipeline/
//	docker run --rm --network none -v $PWD:/p:ro -v <dataset>:/data:ro -v <dir>:/w \
//	  -w /p/internal/pipeline -e TANUH_E2E_PAYLOAD=/w/payload.json -e TANUH_E2E_DATASET=/data \
//	  -e TANUH_E2E_CATALOGUE=/w/item.json -e TANUH_E2E_OUT=/w --entrypoint /w/pipeline.test \
//	  <processing image> -test.run TestLocalEndToEnd -test.v

import (
	"context"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io/fs"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/datakaveri/tanuh-processing-tee/internal/config"
	"github.com/datakaveri/tanuh-processing-tee/internal/crypto"
	"github.com/datakaveri/tanuh-processing-tee/internal/evaluator"
)

func TestLocalEndToEnd(t *testing.T) {
	payloadPath, datasetDir, itemPath := os.Getenv("TANUH_E2E_PAYLOAD"), os.Getenv("TANUH_E2E_DATASET"), os.Getenv("TANUH_E2E_CATALOGUE")
	if payloadPath == "" || datasetDir == "" || itemPath == "" {
		t.Skip("set TANUH_E2E_PAYLOAD, TANUH_E2E_DATASET and TANUH_E2E_CATALOGUE to run a local end-to-end job")
	}
	item, err := os.ReadFile(itemPath)
	if err != nil {
		t.Fatal(err)
	}
	catalogueSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write(item)
	}))
	defer catalogueSrv.Close()

	base := t.TempDir()
	for src, dst := range map[string]string{"../../tools/infer/infer.py": "infer.py", "../../dep_scanner.py": "dep_scanner.py"} {
		b, err := os.ReadFile(src)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(base, dst), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	m, err := NewManager(config.Config{BaseDir: base, CatalogueBaseURL: catalogueSrv.URL,
		DatasetsBucket: "local", DecryptWorkers: 8})
	if err != nil {
		t.Fatal(err)
	}
	m.evaluators = evaluator.ImageSource{Root: "../../evaluators"}
	m.cloud = localEncryptedDataset(t, datasetDir)

	raw, err := os.ReadFile(payloadPath)
	if err != nil {
		t.Fatal(err)
	}
	var payload map[string]any
	if err := json.Unmarshal(raw, &payload); err != nil {
		t.Fatal(err)
	}
	job, err := m.materialize(payload)
	if err != nil {
		t.Fatalf("materialize: %v", err)
	}
	if err := m.runSteps(context.Background(), job); err != nil {
		t.Fatalf("job failed: %v", err)
	}
	results, err := os.ReadFile(filepath.Join(job.runtimeDir, "results.json"))
	if err != nil {
		t.Fatal(err)
	}
	var r map[string]any
	json.Unmarshal(results, &r)
	t.Logf("leaderboard metrics: %v", r["leaderboard_metrics"])
	t.Logf("ground truth: %v", r["ground_truth"])
	t.Logf("stage seconds: %v", r["stage_seconds"])
	if out := os.Getenv("TANUH_E2E_OUT"); out != "" {
		os.WriteFile(filepath.Join(out, "results.json"), results, 0o644)
	}
}

// localEncryptedDataset serves a plaintext folder as the GCS dataset folder
// "<any uuid>/...": every file is an encrypted object plus its manifest,
// encrypted on first download with a fresh key.
func localEncryptedDataset(t *testing.T, root string) cloudAccess {
	t.Helper()
	var rels []string
	err := filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err == nil && !d.IsDir() {
			rel, _ := filepath.Rel(root, p)
			rels = append(rels, filepath.ToSlash(rel))
		}
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	var (
		mu      sync.Mutex
		objects = map[string][]byte{} // object name -> bytes, filled on first download
	)
	encrypt := func(prefix, rel string) error {
		pt, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
		if err != nil {
			return err
		}
		ct, man := encryptDatasetV1(pt, rel)
		mb, _ := json.Marshal(man)
		objects[prefix+rel] = ct
		objects[prefix+rel+manifestSuffix] = mb
		return nil
	}
	return cloudAccess{
		list: func(_ context.Context, _, prefix string) ([]string, error) {
			names := make([]string, 0, 2*len(rels))
			for _, rel := range rels {
				names = append(names, prefix+rel, prefix+rel+manifestSuffix)
			}
			return names, nil
		},
		get: func(_ context.Context, _, object string) ([]byte, error) {
			mu.Lock()
			defer mu.Unlock()
			if b, ok := objects[object]; ok {
				return b, nil
			}
			prefix, rel, _ := strings.Cut(object, "/")
			rel = strings.TrimSuffix(rel, manifestSuffix)
			if err := encrypt(prefix+"/", rel); err != nil {
				return nil, err
			}
			return objects[object], nil
		},
		unwrap: func(_ context.Context, _, wrapped string) ([]byte, error) {
			return base64.StdEncoding.DecodeString(wrapped)
		},
	}
}

// encryptDatasetV1 encrypts like the UI's dataset upload: 1 MiB AES-256-GCM
// chunks, IV = base_iv + chunk index, AAD "index:total:file_id".
func encryptDatasetV1(pt []byte, fileID string) ([]byte, crypto.ManifestV1) {
	const chunk = 1 << 20
	dek, baseIV := make([]byte, 32), make([]byte, 12)
	rand.Read(dek)
	rand.Read(baseIV)
	block, _ := aes.NewCipher(dek)
	aead, _ := cipher.NewGCM(block)
	total := max(1, (len(pt)+chunk-1)/chunk)
	var ct []byte
	for i := 0; i < total; i++ {
		part := pt[min(i*chunk, len(pt)):min((i+1)*chunk, len(pt))]
		iv := append([]byte(nil), baseIV...)
		carry := uint64(i)
		for j := len(iv) - 1; j >= 0 && carry > 0; j-- {
			sum := uint64(iv[j]) + (carry & 0xff)
			iv[j] = byte(sum)
			carry = (carry >> 8) + (sum >> 8)
		}
		ct = append(ct, aead.Seal(nil, iv, part, []byte(fmt.Sprintf("%d:%d:%s", i, total, fileID)))...)
	}
	sum := sha256.Sum256(pt)
	return ct, crypto.ManifestV1{Schema: "tanuh-enc-dataset-v1", FileID: fileID, Cipher: "AES-256-GCM",
		ChunkSize: chunk, TotalChunks: total, PlaintextSize: len(pt), PlaintextSHA256: hex.EncodeToString(sum[:]),
		BaseIV: base64.StdEncoding.EncodeToString(baseIV), WrappedDEK: base64.StdEncoding.EncodeToString(dek),
		KMSKeyVersion: "local"}
}
