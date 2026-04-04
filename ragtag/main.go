package main

import (
	"fmt"
	"os"
	"syscall"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/emirate/rag-tui/model"
)

func main() {
	m := model.New()
	p := tea.NewProgram(
		m,
		tea.WithAltScreen(),
		tea.WithMouseCellMotion(),
	)

	finalModel, err := p.Run()
	if err != nil {
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
