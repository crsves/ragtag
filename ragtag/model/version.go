package model

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	tea "github.com/charmbracelet/bubbletea"
)

// AppVersion is the current binary version — injected at build time via
// -ldflags="-X github.com/emirate/rag-tui/model.AppVersion=vX.Y.Z"
// Falls back to "dev" for local builds.
var AppVersion = "dev"

const versionCheckURL = "https://ragtag.crsv.es/latest_version.json"

// VersionCheckMsg is returned when the background version poll completes.
type VersionCheckMsg struct {
	Latest    string // e.g. "v0.1.4"
	Changelog string
	Err       error
}

// checkVersionCmd polls the mirror for the latest version JSON.
func checkVersionCmd() tea.Cmd {
	return func() tea.Msg {
		client := &http.Client{Timeout: 8 * time.Second}
		resp, err := client.Get(versionCheckURL)
		if err != nil {
			return VersionCheckMsg{Err: err}
		}
		defer resp.Body.Close()
		body, err := io.ReadAll(io.LimitReader(resp.Body, 4096))
		if err != nil {
			return VersionCheckMsg{Err: fmt.Errorf("read: %w", err)}
		}
		var payload struct {
			Version   string `json:"version"`
			Changelog string `json:"changelog"`
		}
		if err := json.Unmarshal(body, &payload); err != nil {
			return VersionCheckMsg{Err: fmt.Errorf("parse: %w", err)}
		}
		return VersionCheckMsg{Latest: payload.Version, Changelog: payload.Changelog}
	}
}

// repeatingVersionCheckCmd sleeps for d then performs a version check.
// The VersionCheckMsg handler should re-schedule this to create a poll loop.
func repeatingVersionCheckCmd(d time.Duration) tea.Cmd {
	return func() tea.Msg {
		time.Sleep(d)
		client := &http.Client{Timeout: 8 * time.Second}
		resp, err := client.Get(versionCheckURL)
		if err != nil {
			return VersionCheckMsg{Err: err}
		}
		defer resp.Body.Close()
		body, err := io.ReadAll(io.LimitReader(resp.Body, 4096))
		if err != nil {
			return VersionCheckMsg{Err: fmt.Errorf("read: %w", err)}
		}
		var payload struct {
			Version   string `json:"version"`
			Changelog string `json:"changelog"`
		}
		if err := json.Unmarshal(body, &payload); err != nil {
			return VersionCheckMsg{Err: fmt.Errorf("parse: %w", err)}
		}
		return VersionCheckMsg{Latest: payload.Version, Changelog: payload.Changelog}
	}
}

// isNewerVersion returns true if remote is a newer semver than local.
// Handles "v0.1.4" style strings.
func isNewerVersion(local, remote string) bool {
	if local == "dev" || remote == "" {
		return false
	}
	// Simple string compare works for semver with same prefix length and zero-padded.
	// For proper semver we do a numeric field compare.
	return semverGreater(remote, local)
}

func semverGreater(a, b string) bool {
	aMaj, aMin, aPatch := parseSemver(a)
	bMaj, bMin, bPatch := parseSemver(b)
	if aMaj != bMaj {
		return aMaj > bMaj
	}
	if aMin != bMin {
		return aMin > bMin
	}
	return aPatch > bPatch
}

func parseSemver(v string) (int, int, int) {
	// Strip leading 'v'
	if len(v) > 0 && v[0] == 'v' {
		v = v[1:]
	}
	var maj, min, patch int
	fmt.Sscanf(v, "%d.%d.%d", &maj, &min, &patch)
	return maj, min, patch
}
