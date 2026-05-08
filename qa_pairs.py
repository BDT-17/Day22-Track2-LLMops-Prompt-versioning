QA_PAIRS = [
    {
        "question": "What are the three main types of machine learning?",
        "reference": "The three main types of machine learning are supervised learning, unsupervised learning, and reinforcement learning.",
    },
    {
        "question": "What is overfitting in machine learning?",
        "reference": "Overfitting occurs when a model learns the training data too well, including noise, leading to poor generalization on new data.",
    },
    {
        "question": "Explain the bias-variance tradeoff.",
        "reference": "High bias means underfitting, while high variance means overfitting. Good models balance both to minimize total error.",
    },
    {
        "question": "How does regularization prevent overfitting?",
        "reference": "Regularization adds penalties such as L1 or L2 terms to discourage overly complex models and reduce overfitting.",
    },
    {
        "question": "What is cross-validation?",
        "reference": "Cross-validation splits data into multiple folds so model performance can be estimated more reliably across repeated train and validation runs.",
    },
    {
        "question": "What is backpropagation?",
        "reference": "Backpropagation computes gradients of the loss with respect to network weights by applying the chain rule from the output layer back through the network.",
    },
    {
        "question": "What are Convolutional Neural Networks primarily used for?",
        "reference": "Convolutional Neural Networks are primarily used for grid-like data such as images because they can learn local spatial patterns.",
    },
    {
        "question": "How do LSTM networks address the vanishing gradient problem?",
        "reference": "LSTM networks use gates and a memory cell to preserve useful information over longer sequences, which helps reduce vanishing gradients.",
    },
    {
        "question": "What activation functions are commonly used in neural networks?",
        "reference": "Common activation functions include ReLU, sigmoid, tanh, and sometimes GELU in modern transformer models.",
    },
    {
        "question": "What is the role of pooling layers in CNNs?",
        "reference": "Pooling layers reduce spatial dimensions, lower computation, and keep the most important local features in CNNs.",
    },
    {
        "question": "What is the transformer architecture?",
        "reference": "The transformer architecture uses self-attention and feed-forward layers to process sequences in parallel instead of relying on recurrence.",
    },
    {
        "question": "What are word embeddings?",
        "reference": "Word embeddings are dense vector representations that place semantically similar words near each other in vector space.",
    },
    {
        "question": "What is transfer learning in NLP?",
        "reference": "Transfer learning in NLP means pretraining a language model on large corpora and then fine-tuning it on specific downstream tasks.",
    },
    {
        "question": "How does BERT handle language understanding?",
        "reference": "BERT uses bidirectional transformer training with masked language modeling so it can learn context from both the left and right side of a token.",
    },
    {
        "question": "What is self-attention in transformers?",
        "reference": "Self-attention lets each token weigh the relevance of other tokens in the sequence when building its contextual representation.",
    },
    {
        "question": "What is GPT and how is it trained?",
        "reference": "GPT is an autoregressive transformer trained to predict the next token from previous tokens using large text corpora.",
    },
    {
        "question": "What is instruction tuning?",
        "reference": "Instruction tuning fine-tunes a pretrained language model on instruction-response examples so it follows user intent more effectively.",
    },
    {
        "question": "What is RLHF?",
        "reference": "RLHF stands for Reinforcement Learning from Human Feedback and uses preference data to align model behavior with human expectations.",
    },
    {
        "question": "What is chain-of-thought prompting?",
        "reference": "Chain-of-thought prompting encourages a model to reason through intermediate steps, which can improve performance on complex tasks.",
    },
    {
        "question": "What is the context length of GPT-4?",
        "reference": "Some GPT-4 variants support up to 128K tokens of context, though exact context length depends on the specific model version.",
    },
    {
        "question": "What is Retrieval-Augmented Generation?",
        "reference": "Retrieval-Augmented Generation combines an LLM with retrieval from an external knowledge source so answers stay grounded in relevant context.",
    },
    {
        "question": "What are the main components of a RAG pipeline?",
        "reference": "A RAG pipeline usually includes document ingestion, chunking, embeddings, a vector index, retrieval, prompt construction, and answer generation.",
    },
    {
        "question": "What is dense retrieval?",
        "reference": "Dense retrieval represents queries and documents with embeddings and measures semantic similarity in vector space to find relevant documents.",
    },
    {
        "question": "Why is chunking strategy important in RAG?",
        "reference": "Chunking strategy affects retrieval quality because chunks must be small enough to stay focused but large enough to preserve the needed context.",
    },
    {
        "question": "What advanced RAG techniques exist beyond basic retrieval?",
        "reference": "Advanced RAG techniques include reranking, query expansion, hypothetical document embeddings, multi-hop retrieval, and hybrid search.",
    },
    {
        "question": "What are vector databases used for?",
        "reference": "Vector databases store embeddings and support fast similarity search for semantic retrieval tasks.",
    },
    {
        "question": "What is FAISS?",
        "reference": "FAISS is a similarity search library from Meta that supports efficient nearest-neighbor retrieval over dense vectors.",
    },
    {
        "question": "How do text embeddings capture semantic meaning?",
        "reference": "Text embeddings map text into numeric vectors where semantically similar texts are positioned closer together in vector space.",
    },
    {
        "question": "What is HNSW?",
        "reference": "HNSW is a graph-based approximate nearest neighbor algorithm that builds a hierarchy of navigable small-world graphs for efficient search.",
    },
    {
        "question": "What is hybrid search in vector databases?",
        "reference": "Hybrid search combines dense semantic retrieval with sparse keyword methods such as BM25 to improve relevance.",
    },
    {
        "question": "What is LangChain?",
        "reference": "LangChain is a framework for building LLM applications using abstractions for prompts, models, retrieval, tools, and orchestration.",
    },
    {
        "question": "What is LangChain Expression Language (LCEL)?",
        "reference": "LCEL is LangChain's compositional syntax that uses the pipe operator to build chains from reusable runnables.",
    },
    {
        "question": "What is LangGraph?",
        "reference": "LangGraph extends LangChain with graph-based orchestration for stateful, multi-step, and cyclic LLM workflows.",
    },
    {
        "question": "What memory types does LangChain support?",
        "reference": "LangChain supports several memory patterns, including buffer memory, summary memory, windowed memory, and vector-store-backed memory.",
    },
    {
        "question": "What are LangChain retrievers?",
        "reference": "LangChain retrievers fetch relevant documents from a knowledge source given a query, often using vector stores or keyword search.",
    },
    {
        "question": "What is LangSmith?",
        "reference": "LangSmith is a platform for tracing, debugging, evaluating, and monitoring LLM applications.",
    },
    {
        "question": "What information do LangSmith traces capture?",
        "reference": "LangSmith traces can capture inputs, outputs, intermediate steps, latency, token usage, and errors across an LLM workflow.",
    },
    {
        "question": "What is the LangSmith Prompt Hub?",
        "reference": "The LangSmith Prompt Hub stores and versions prompt templates so teams can share, compare, and roll back prompts.",
    },
    {
        "question": "How does LangSmith help monitor production LLM applications?",
        "reference": "LangSmith helps monitor production systems with traces, latency metrics, debugging details, evaluations, and feedback workflows.",
    },
    {
        "question": "What are LangSmith datasets used for?",
        "reference": "LangSmith datasets are used to store evaluation examples so prompts and models can be tested consistently across versions.",
    },
    {
        "question": "What is RAGAS?",
        "reference": "RAGAS is a framework for evaluating retrieval-augmented generation systems using metrics such as faithfulness and answer relevancy.",
    },
    {
        "question": "How does RAGAS compute faithfulness?",
        "reference": "RAGAS faithfulness measures whether the claims made in an answer can be supported by the retrieved context.",
    },
    {
        "question": "What is answer relevancy in RAGAS?",
        "reference": "Answer relevancy measures how well an answer addresses the user's original question.",
    },
    {
        "question": "What is context recall in RAGAS?",
        "reference": "Context recall measures whether the retrieved passages cover the information needed to support the reference answer.",
    },
    {
        "question": "What inputs does RAGAS evaluation require?",
        "reference": "RAGAS evaluation typically uses the user question, generated answer, retrieved contexts, and often a reference answer.",
    },
    {
        "question": "What is Guardrails AI?",
        "reference": "Guardrails AI is a framework for validating and correcting LLM outputs with custom or reusable validators and configurable failure actions.",
    },
    {
        "question": "What is PII and why is it important to detect in LLM responses?",
        "reference": "PII is personally identifiable information, and detecting it matters because leaking it can create privacy, compliance, and security risks.",
    },
    {
        "question": "What does structured output validation ensure?",
        "reference": "Structured output validation ensures model responses match the expected schema or format, such as valid JSON.",
    },
    {
        "question": "What is Constitutional AI?",
        "reference": "Constitutional AI is an alignment technique where a model follows a set of principles and uses self-critique to improve responses.",
    },
    {
        "question": "What are common AI safety concerns with LLMs?",
        "reference": "Common AI safety concerns include hallucination, bias, toxic output, privacy leakage, prompt injection, and jailbreaking.",
    },
]

SAMPLE_QUESTIONS = [item["question"] for item in QA_PAIRS]
