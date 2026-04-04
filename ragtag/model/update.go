package model

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"runtime"

	tea "github.com/charmbracelet/bubbletea"
)

const (
	mirrorBase  = "https://ragtag.crsv.es/releases/latest/download"
	githubBase  = "https://github.com/crsves/ragtag/releases/latest/download"
)

// binaryName returns the release asset name for the running OS/arch.
func binaryName() (string, error) {
	goos := runtime.GOOS
	goarch := runtime.GOARCH

	switch goos {
	case "darwin":
		if goarch == "arm64" {
			return "ragtag-mac-arm64", nil
		}
		return "ragtag-mac-intel", nil
	case "linux":
		switch goarch {
		case "amd64":
			return "ragtag-linux-amd64", nil
		case "arm64":
			return "ragtag-linux-arm64", nil
		}
	}
	return "", fmt.Errorf("unsupported platform: %s/%s", goos, goarch)
}

// selfUpdateCmd downloads the latest binary and replaces the current executable.
func selfUpdateCmd() tea.Cmd {
	return func() tea.Msg {
		name, err := binaryName()
		if err != nil {
			return UpdateMsg{Err: err}
		}

		// Try mirror first, then GitHub.
		urls := []string{
			mirrorBase + "/" + name,
			githubBase + "/" + name,
		}

		var body []byte
		var lastErr error
		for _, url := range urls {
			body, lastErr = downloadBinary(url)
			if lastErr == nil {
				break
			}
		}
		if lastErr != nil {
			return UpdateMsg{Err: fmt.Errorf("download failed: %w", lastErr)}
		}

		// Find where we're running from.
		exePath, err := os.Executable()
		if err != nil {
			return UpdateMsg{Err: fmt.Errorf("could not resolve executable path: %w", err)}
		}

		// Write to a temp file next to the current binary for atomic rename.
		tmp, err := os.CreateTemp("", "ragtag-update-*")
		if err != nil {
			return UpdateMsg{Err: fmt.Errorf("could not create temp file: %w", err)}
		}
		tmpPath := tmp.Name()

		if _, err = tmp.Write(body); err != nil {
			tmp.Close()
			os.Remove(tmpPath)
			return UpdateMsg{Err: fmt.Errorf("could not write temp file: %w", err)}
		}
		tmp.Close()

		// Make it executable.
		if err = os.Chmod(tmpPath, 0755); err != nil {
			os.Remove(tmpPath)
			return UpdateMsg{Err: fmt.Errorf("chmod failed: %w", err)}
		}

		// Atomic replace.
		if err = os.Rename(tmpPath, exePath); err != nil {
			os.Remove(tmpPath)
			return UpdateMsg{Err: fmt.Errorf("could not replace binary: %w", err)}
		}

		return UpdateMsg{}
	}
}

func downloadBinary(url string) ([]byte, error) {
	resp, err := http.Get(url) //nolint:gosec
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d from %s", resp.StatusCode, url)
	}

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	// Sanity check: binary must be at least 1 MB.
	if len(data) < 1<<20 {
		return nil, fmt.Errorf("download too small (%d bytes) — likely a stub or error page", len(data))
	}

	return data, nil
}
