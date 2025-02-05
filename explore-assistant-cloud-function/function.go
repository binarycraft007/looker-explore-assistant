package function

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"os"

	"github.com/GoogleCloudPlatform/functions-framework-go/functions"
	"google.golang.org/genai"
)

var (
	project           = os.Getenv("PROJECT")
	location          = os.Getenv("REGION")
	vertexCFAuthToken = os.Getenv("VERTEX_CF_AUTH_TOKEN")
	modelName         = os.Getenv("MODEL_NAME")
	ragCorpus         = os.Getenv("RAG_CORPUS")
)

type RequestBody struct {
	Contents   string                 `json:"contents"`
	Parameters map[string]interface{} `json:"parameters,omitempty"`
}

func getResponseHeaders(w http.ResponseWriter) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, X-Signature")
}

func hasValidSignature(r *http.Request, body []byte) bool {
	signature := r.Header.Get("X-Signature")
	if signature == "" {
		return false
	}

	h := hmac.New(sha256.New, []byte(vertexCFAuthToken))
	h.Write(body)
	expectedSignature := hex.EncodeToString(h.Sum(nil))

	return hmac.Equal([]byte(signature), []byte(expectedSignature))
}

func generateLookerQuery(contents string, parameters map[string]interface{}) (string, error) {
	_ = parameters
	ctx := context.Background()
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		Project:  project,
		Location: location,
		Backend:  genai.BackendVertexAI,
	})
	if err != nil {
		return "", err
	}

	config := &genai.GenerateContentConfig{
		Temperature:     genai.Ptr(float64(0.2)),
		TopP:            genai.Ptr(float64(0.8)),
		TopK:            genai.Ptr(float64(40)),
		MaxOutputTokens: genai.Ptr(int64(500)),
		CandidateCount:  genai.Ptr(int64(1)),
		Tools: []*genai.Tool{
			{
				Retrieval: &genai.Retrieval{
					VertexRAGStore: &genai.VertexRAGStore{
						RAGResources: []*genai.VertexRAGStoreRAGResource{
							{RAGCorpus: ragCorpus},
						},
					},
				},
			},
		},
	}
	response, err := client.Models.GenerateContent(ctx, modelName, genai.Text(contents), config)
	if err != nil {
		return "", err
	}

	return response.Text()
}

func HandleRequest(w http.ResponseWriter, r *http.Request) {
	getResponseHeaders(w)

	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusNoContent)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Failed to read request body", http.StatusInternalServerError)
		return
	}

	if !hasValidSignature(r, body) {
		http.Error(w, "Invalid signature", http.StatusForbidden)
		return
	}

	var req RequestBody
	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	if req.Contents == "" {
		http.Error(w, "Missing 'contents' parameter", http.StatusBadRequest)
		return
	}

	responseText, err := generateLookerQuery(req.Contents, req.Parameters)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{"response": responseText})
}

func init() {
	functions.HTTP("HandleRequest", HandleRequest)
}
