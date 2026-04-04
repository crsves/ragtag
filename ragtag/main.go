package main

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime/debug"
	"syscall"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/emirate/rag-tui/model"
)

// crashLog writes a message to ~/ragtag/crash.log (or the ragDir equivalent).
func crashLog(msg string) {
	home, _ := os.UserHomeDir()
	for _, dir := range []string{
		filepath.Join(home, "raa"),
		filepath.Join(home, "ragtag"),
	} {
		if fi, err := os.Stat(dir); err == nil && fi.IsDir() {
			f, err := os.OpenFile(filepath.Join(dir, "crash.log"), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
			if err == nil {
				fmt.Fprintf(f, "[%s] %s\n", time.Now().Format(time.RFC3339), msg)
				f.Close()
			}
			return
		}
	}
}

func main() {
	defer func() {
		if r := recover(); r != nil {
			crashLog(fmt.Sprintf("PANIC in main: %v\n%s", r, debug.Stack()))
			fmt.Fprintf(os.Stderr, "ragtag panic: %v\n", r)
			os.Exit(2)
		}
	}()

	m := model.New()
	p := tea.NewProgram(
		m,
		tea.WithAltScreen(),
		tea.WithMouseCellMotion(),
	)

	finalModel, err := p.Run()
	if err != nil {
		crashLog(fmt.Sprintf("bubbletea error: %v", err))
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	if am, ok := finalModel.(model.AppModel); ok && am.ShouldRestart() {
		exe, err := os.Executable()
		if err != nil {
			fmt.Fprintln(os.Stderr, "restart: could not locate binary:", err)
			os.Exit(1)
		}
		_ = syscall.Exec(exe, os.Args, os.Environ())
	}
}
