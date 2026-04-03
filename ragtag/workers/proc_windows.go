//go:build windows

package workers

import (
	"os/exec"
	"syscall"
)

// setProcAttr creates the subprocess in a new process group on Windows so it
// can be reliably killed without leaving orphans in Task Manager.
func setProcAttr(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{
		CreationFlags: syscall.CREATE_NEW_PROCESS_GROUP,
	}
}

// killProc kills the bridge process on Windows.
func killProc(cmd *exec.Cmd) {
	if cmd == nil || cmd.Process == nil {
		return
	}
	_ = cmd.Process.Kill()
}
