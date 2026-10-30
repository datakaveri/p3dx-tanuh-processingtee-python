package eval

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"time"
)

// DepsError reports a failed dependency install for one of the model
// provider's scripts (dep_scanner.py + uv). Output carries the scanner's
// stderr tail for classification/logging; it is sanitised before any external
// reporting.
type DepsError struct {
	Script string // "adaptor.py" or "preprocessing.py"
	Output string
}

func (e *DepsError) Error() string {
	script := e.Script
	if script == "" {
		script = "adaptor.py"
	}
	return "installing " + script + " dependencies failed: " + e.Output
}

// InstallDeps runs dep_scanner.py against one of the model provider's scripts
// (adaptor.py, preprocessing.py): it AST-scans the imports and uv-installs
// whatever the image doesn't already provide, so the stage doesn't die on
// ImportError. A failed install fails the job
// with the uv output rather than a confusing ImportError later.
func InstallDeps(ctx context.Context, workdir, scannerPath, scriptPath string, timeout time.Duration) error {
	if timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, timeout)
		defer cancel()
	}

	log.Printf("eval: scanning %s for missing dependencies", scriptPath)
	cmd := exec.CommandContext(ctx, "python3", scannerPath, scriptPath)
	cmd.Dir = workdir

	var stderr bytes.Buffer
	cmd.Stdout = os.Stdout
	cmd.Stderr = io.MultiWriter(os.Stderr, &stderr)

	err := cmd.Run()
	if err == nil {
		return nil
	}
	script := filepath.Base(scriptPath)
	if ctx.Err() == context.DeadlineExceeded {
		return &DepsError{Script: script, Output: fmt.Sprintf("dependency install timed out after %s", timeout)}
	}
	tail := stderr.String()
	if len(tail) > 1024 {
		tail = tail[len(tail)-1024:]
	}
	return &DepsError{Script: script, Output: tail}
}
