//go:build !windows

package workers

import (
	"os/exec"
	"syscall"
)

// setProcAttr puts the subprocess in its own process group so a Kill()
// on the group terminates all child processes cleanly on Unix/macOS.
func setProcAttr(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
}

// killProc kills the bridge process and its whole process group.
func killProc(cmd *exec.Cmd) {
	if cmd == nil || cmd.Process == nil {
		return
	}
	// Kill the entire process group to avoid orphan children.
	pgid := -cmd.Process.Pid
	_ = syscall.Kill(pgid, syscall.SIGKILL)
	_ = cmd.Process.Kill() // belt-and-suspenders
}
