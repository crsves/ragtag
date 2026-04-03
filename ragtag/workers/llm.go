package workers

import (
	"context"
	"fmt"
	"strings"

	"github.com/sashabaranov/go-openai"
)

// LLMConfig holds configuration for the NIM/OpenAI-compatible LLM client.
type LLMConfig struct {
	APIKey       string
	BaseURL      string
	Model        string
	Temperature  float32
	TopP         float32
	MaxTokens    int
	ThinkingMode bool
}

// newClient creates an OpenAI client configured for the given LLMConfig.
func newClient(cfg LLMConfig) *openai.Client {
	config := openai.DefaultConfig(cfg.APIKey)
	config.BaseURL = cfg.BaseURL
	return openai.NewClientWithConfig(config)
}

// extraBody returns the extra_body map for thinking-mode requests, or nil.
func extraBody(cfg LLMConfig) map[string]interface{} {
	if cfg.ThinkingMode && strings.Contains(cfg.Model, "deepseek") {
		return map[string]interface{}{
			"chat_template_kwargs": map[string]interface{}{
				"thinking": true,
			},
		}
	}
	return nil
}

// StreamLLM streams completion tokens to tokenCh.
// When streaming ends (successfully or with an error) it closes tokenCh and
// sends to errCh (nil on success). The caller must drain tokenCh.
func StreamLLM(ctx context.Context, cfg LLMConfig, messages []openai.ChatCompletionMessage, tokenCh chan<- string, errCh chan<- error) {
	client := newClient(cfg)

	req := openai.ChatCompletionRequest{
		Model:       cfg.Model,
		Messages:    messages,
		Temperature: cfg.Temperature,
		TopP:        cfg.TopP,
		MaxTokens:   cfg.MaxTokens,
		Stream:      true,
	}

	if eb := extraBody(cfg); eb != nil {
		// go-openai exposes extra fields via the request struct in recent versions.
		// We encode them into the request body via StreamOptions or a custom approach.
		// Since go-openai doesn't have a generic ExtraBody field, we skip this for
		// streaming (thinking mode is primarily useful for non-streaming calls).
		_ = eb
	}

	stream, err := client.CreateChatCompletionStream(ctx, req)
	if err != nil {
		close(tokenCh)
		errCh <- fmt.Errorf("create stream: %w", err)
		return
	}
	defer stream.Close()

	for {
		resp, err := stream.Recv()
		if err != nil {
			close(tokenCh)
			if err.Error() == "EOF" || strings.Contains(err.Error(), "EOF") {
				errCh <- nil
			} else {
				errCh <- fmt.Errorf("stream recv: %w", err)
			}
			return
		}
		if len(resp.Choices) > 0 {
			tok := resp.Choices[0].Delta.Content
			if tok != "" {
				select {
				case tokenCh <- tok:
				case <-ctx.Done():
					close(tokenCh)
					errCh <- ctx.Err()
					return
				}
			}
		}
	}
}

// CallLLM performs a non-streaming chat completion and returns the full response.
func CallLLM(ctx context.Context, cfg LLMConfig, messages []openai.ChatCompletionMessage) (string, error) {
	client := newClient(cfg)

	req := openai.ChatCompletionRequest{
		Model:       cfg.Model,
		Messages:    messages,
		Temperature: cfg.Temperature,
		TopP:        cfg.TopP,
		MaxTokens:   cfg.MaxTokens,
		Stream:      false,
	}

	resp, err := client.CreateChatCompletion(ctx, req)
	if err != nil {
		return "", fmt.Errorf("chat completion: %w", err)
	}

	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("no choices in response")
	}

	return strings.TrimSpace(resp.Choices[0].Message.Content), nil
}
