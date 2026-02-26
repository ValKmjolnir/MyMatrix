/* word2vec.hpp - Skip-gram Word2Vec Implementation */
/* by ValKmjolnir 2026/02/27 */

#pragma once

#include "matrix.hpp"

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <cmath>
#include <cstdint>

template<typename T = float>
class Word2Vec {
    static_assert(std::is_floating_point<T>::value, "T must be floating point type");

public:
    struct TrainingConfig {
        size_t embedding_dim = 100;
        size_t window_size = 5;
        size_t negative_samples = 5;
        T learning_rate = 0.025f;
        size_t epochs = 10;
        T min_count = 1;
        T subsample_threshold = 1e-3f;
        bool use_negative_sampling = true;
    };

private:
    std::unordered_map<std::string, size_t> word2idx;
    std::vector<std::string> idx2word;
    std::unordered_map<std::string, size_t> word_counts;
    size_t vocab_size = 0;
    size_t total_words = 0;

    matrix<T>* word_embeddings = nullptr;  // Input embeddings (W)
    matrix<T>* context_embeddings = nullptr;  // Output embeddings (W')

    TrainingConfig config;
    std::mt19937 rng;

    std::vector<std::string> corpus;
    std::vector<std::vector<size_t>> sentences;

private:
    void build_vocabulary() {
        word2idx.clear();
        idx2word.clear();
        vocab_size = 0;

        for (const auto& word : corpus) {
            if (word_counts[word] >= config.min_count) {
                if (word2idx.find(word) == word2idx.end()) {
                    word2idx[word] = vocab_size;
                    idx2word.push_back(word);
                    vocab_size++;
                }
            }
        }
    }

    T get_subsample_prob(const std::string& word) const {
        if (total_words == 0) return 0.0f;
        T freq = static_cast<T>(word_counts.at(word)) / static_cast<T>(total_words);
        return 1.0f - std::sqrt(config.subsample_threshold / freq);
    }

    std::vector<size_t> generate_negative_samples(size_t target_idx, size_t n_samples) {
        std::vector<size_t> negatives;
        std::uniform_int_distribution<size_t> dist(0, vocab_size - 1);

        for (size_t i = 0; i < n_samples; ++i) {
            size_t neg_idx;
            do {
                neg_idx = dist(rng);
            } while (neg_idx == target_idx);
            negatives.push_back(neg_idx);
        }
        return negatives;
    }

public:
    Word2Vec() : rng(std::random_device{}()) {}

    Word2Vec(const TrainingConfig& cfg) : config(cfg), rng(std::random_device{}()) {}

    ~Word2Vec() {
        if (word_embeddings) delete word_embeddings;
        if (context_embeddings) delete context_embeddings;
    }

    void load_corpus(const std::string& text) {
        corpus.clear();
        std::istringstream iss(text);
        std::string word;

        while (iss >> word) {
            // Simple tokenization: convert to lowercase and remove non-alpha
            std::string cleaned;
            for (char c : word) {
                if (std::isalpha(static_cast<unsigned char>(c))) {
                    cleaned += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                }
            }
            if (!cleaned.empty()) {
                corpus.push_back(cleaned);
                word_counts[cleaned]++;
                total_words++;
            }
        }
    }

    void load_corpus_from_file(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filepath);
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        load_corpus(buffer.str());
    }

    void prepare_training_data() {
        build_vocabulary();

        // Apply subsampling and convert to indices
        sentences.clear();
        std::vector<size_t> current_sentence;

        for (const auto& word : corpus) {
            if (word2idx.find(word) == word2idx.end()) continue;

            // Subsampling
            T prob = get_subsample_prob(word);
            std::uniform_real_distribution<T> dist(0.0f, 1.0f);

            if (prob > dist(rng)) {
                current_sentence.push_back(word2idx[word]);
            } else {
                if (!current_sentence.empty()) {
                    sentences.push_back(current_sentence);
                    current_sentence.clear();
                }
            }
        }
        if (!current_sentence.empty()) {
            sentences.push_back(current_sentence);
        }
    }

    void initialize_embeddings() {
        if (vocab_size == 0) {
            throw std::runtime_error("Vocabulary not built. Call prepare_training_data() first.");
        }

        if (word_embeddings) delete word_embeddings;
        if (context_embeddings) delete context_embeddings;

        word_embeddings = new matrix<T>(vocab_size, config.embedding_dim);
        context_embeddings = new matrix<T>(vocab_size, config.embedding_dim);

        // Initialize with small random values
        std::uniform_real_distribution<T> dist(-0.5f / config.embedding_dim, 0.5f / config.embedding_dim);

        for (size_t i = 0; i < vocab_size; ++i) {
            for (size_t j = 0; j < config.embedding_dim; ++j) {
                (*word_embeddings)[i][j] = dist(rng);
                (*context_embeddings)[i][j] = dist(rng);
            }
        }
    }

    void train() {
        if (!word_embeddings || !context_embeddings) {
            initialize_embeddings();
        }

        std::uniform_int_distribution<size_t> neg_dist(0, vocab_size - 1);

        for (size_t epoch = 0; epoch < config.epochs; ++epoch) {
            T total_loss = 0.0f;
            size_t total_pairs = 0;

            for (const auto& sentence : sentences) {
                for (size_t pos = 0; pos < sentence.size(); ++pos) {
                    size_t target_idx = sentence[pos];

                    // Generate context window
                    size_t window_start = (pos > config.window_size) ? (pos - config.window_size) : 0;
                    size_t window_end = std::min(pos + config.window_size + 1, sentence.size());

                    for (size_t ctx_pos = window_start; ctx_pos < window_end; ++ctx_pos) {
                        if (ctx_pos == pos) continue;

                        size_t context_idx = sentence[ctx_pos];

                        if (config.use_negative_sampling) {
                            // Negative sampling loss
                            auto neg_samples = generate_negative_samples(target_idx, config.negative_samples);

                            // Positive sample gradient
                            matrix<T> target_emb(1, config.embedding_dim);
                            matrix<T> context_emb(1, config.embedding_dim);

                            for (size_t d = 0; d < config.embedding_dim; ++d) {
                                target_emb[0][d] = (*word_embeddings)[target_idx][d];
                                context_emb[0][d] = (*context_embeddings)[context_idx][d];
                            }

                            T dot_product = 0.0f;
                            for (size_t d = 0; d < config.embedding_dim; ++d) {
                                dot_product += target_emb[0][d] * context_emb[0][d];
                            }

                            T sigmoid_val = 1.0f / (1.0f + std::exp(-dot_product));
                            T grad = config.learning_rate * (1.0f - sigmoid_val);

                            // Update embeddings
                            for (size_t d = 0; d < config.embedding_dim; ++d) {
                                (*word_embeddings)[target_idx][d] += grad * context_emb[0][d];
                                (*context_embeddings)[context_idx][d] += grad * target_emb[0][d];
                            }

                            total_loss -= std::log(sigmoid_val + 1e-10f);

                            // Negative samples
                            for (size_t neg_idx : neg_samples) {
                                for (size_t d = 0; d < config.embedding_dim; ++d) {
                                    context_emb[0][d] = (*context_embeddings)[neg_idx][d];
                                }

                                dot_product = 0.0f;
                                for (size_t d = 0; d < config.embedding_dim; ++d) {
                                    dot_product += target_emb[0][d] * context_emb[0][d];
                                }

                                sigmoid_val = 1.0f / (1.0f + std::exp(-dot_product));
                                grad = config.learning_rate * (0.0f - sigmoid_val);

                                for (size_t d = 0; d < config.embedding_dim; ++d) {
                                    (*word_embeddings)[target_idx][d] += grad * context_emb[0][d];
                                    (*context_embeddings)[neg_idx][d] += grad * target_emb[0][d];
                                }

                                total_loss -= std::log(1.0f - sigmoid_val + 1e-10f);
                            }

                            total_pairs += (1 + config.negative_samples);
                        }
                    }
                }
            }

            T avg_loss = (total_pairs > 0) ? total_loss / total_pairs : 0.0f;
            std::cout << "Epoch " << (epoch + 1) << "/" << config.epochs
                      << " - Loss: " << avg_loss << std::endl;
        }
    }

    std::vector<T> get_word_vector(const std::string& word) const {
        auto it = word2idx.find(word);
        if (it == word2idx.end()) {
            throw std::runtime_error("Word not in vocabulary: " + word);
        }

        size_t idx = it->second;
        std::vector<T> vec(config.embedding_dim);
        for (size_t j = 0; j < config.embedding_dim; ++j) {
            vec[j] = (*word_embeddings)[idx][j];
        }
        return vec;
    }

    std::vector<std::pair<std::string, T>> most_similar(const std::string& word, size_t top_n = 10) const {
        auto target_vec = get_word_vector(word);

        // Normalize target vector
        T norm = 0.0f;
        for (size_t i = 0; i < config.embedding_dim; ++i) {
            norm += target_vec[i] * target_vec[i];
        }
        norm = std::sqrt(norm);
        if (norm > 0) {
            for (size_t i = 0; i < config.embedding_dim; ++i) {
                target_vec[i] /= norm;
            }
        }

        std::vector<std::pair<std::string, T>> similarities;

        for (size_t idx = 0; idx < vocab_size; ++idx) {
            if (idx2word[idx] == word) continue;

            T dot = 0.0f;
            T vec_norm = 0.0f;

            for (size_t d = 0; d < config.embedding_dim; ++d) {
                dot += target_vec[d] * (*word_embeddings)[idx][d];
                vec_norm += (*word_embeddings)[idx][d] * (*word_embeddings)[idx][d];
            }
            vec_norm = std::sqrt(vec_norm);

            T cosine_sim = (vec_norm > 0) ? dot / vec_norm : 0.0f;
            similarities.emplace_back(idx2word[idx], cosine_sim);
        }

        // Sort by similarity
        std::sort(similarities.begin(), similarities.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        if (similarities.size() > top_n) {
            similarities.resize(top_n);
        }

        return similarities;
    }

    size_t get_vocab_size() const { return vocab_size; }
    size_t get_embedding_dim() const { return config.embedding_dim; }

    void save_embeddings(const std::string& filepath) const {
        // Save vocabulary to text file
        std::string vocab_path = filepath + ".vocab";
        std::ofstream vocab_out(vocab_path);
        if (!vocab_out.is_open()) {
            throw std::runtime_error("Cannot open file for writing: " + vocab_path);
        }

        vocab_out << vocab_size << " " << config.embedding_dim << "\n";
        for (size_t i = 0; i < vocab_size; ++i) {
            vocab_out << idx2word[i];
            for (size_t j = 0; j < config.embedding_dim; ++j) {
                vocab_out << " " << (*word_embeddings)[i][j];
            }
            vocab_out << "\n";
        }
        vocab_out.close();

        // Save context embeddings to binary file (using matrix::save)
        std::string weight_path = filepath + ".weights";
        std::ofstream weight_out(weight_path, std::ios::binary);
        if (!weight_out.is_open()) {
            throw std::runtime_error("Cannot open file for writing: " + weight_path);
        }
        context_embeddings->save(weight_out);
        weight_out.close();
    }

    void load_embeddings(const std::string& filepath) {
        // Load vocabulary and word embeddings from text file
        std::string vocab_path = filepath + ".vocab";
        std::ifstream vocab_in(vocab_path);
        if (!vocab_in.is_open()) {
            throw std::runtime_error("Cannot open file for reading: " + vocab_path);
        }

        idx2word.clear();
        word2idx.clear();

        size_t saved_vocab_size, saved_dim;
        vocab_in >> saved_vocab_size >> saved_dim;

        if (word_embeddings) delete word_embeddings;
        word_embeddings = new matrix<T>(saved_vocab_size, saved_dim);

        for (size_t i = 0; i < saved_vocab_size; ++i) {
            std::string word;
            vocab_in >> word;
            idx2word.push_back(word);
            word2idx[word] = i;
            for (size_t j = 0; j < saved_dim; ++j) {
                vocab_in >> (*word_embeddings)[i][j];
            }
        }
        vocab_in.close();

        // Load context embeddings from binary file
        std::string weight_path = filepath + ".weights";
        std::ifstream weight_in(weight_path, std::ios::binary);
        if (!weight_in.is_open()) {
            throw std::runtime_error("Cannot open file for reading: " + weight_path);
        }

        if (context_embeddings) delete context_embeddings;
        context_embeddings = new matrix<T>(1, 1);
        context_embeddings->load(weight_in);
        weight_in.close();

        vocab_size = saved_vocab_size;
        config.embedding_dim = saved_dim;
    }

    void save_text_format(const std::string& filepath) const {
        std::ofstream out(filepath);
        if (!out.is_open()) {
            throw std::runtime_error("Cannot open file for writing: " + filepath);
        }

        out << vocab_size << " " << config.embedding_dim << "\n";

        for (size_t i = 0; i < vocab_size; ++i) {
            out << idx2word[i];
            for (size_t j = 0; j < config.embedding_dim; ++j) {
                out << " " << (*word_embeddings)[i][j];
            }
            out << "\n";
        }

        out.close();
    }
};
