/* word2vec.cpp - Skip-gram Word2Vec Demo */
/* by ValKmjolnir 2026/02/27 */

#include "word2vec.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>

void demo_with_sample_text() {
    std::cout << "=== Word2Vec Skip-gram Demo ===" << std::endl;

    // Sample corpus for demonstration
    std::string sample_text = R"(
        the quick brown fox jumps over the lazy dog
        the dog barks at the fox
        the fox runs through the forest
        the brown dog chases the quick fox
        machine learning is a subset of artificial intelligence
        deep learning uses neural networks for learning
        natural language processing enables computers to understand text
        word embeddings capture semantic relationships between words
        the cat sits on the mat
        the dog plays with the cat
        artificial intelligence will transform many industries
        neural networks are inspired by biological neurons
        the quick brown fox and the lazy dog are friends
        learning representations is key to modern ai
        words with similar meanings have similar vectors
    )";

    Word2Vec<float>::TrainingConfig config;
    config.embedding_dim = 128;
    config.window_size = 5;
    config.negative_samples = 5;
    config.learning_rate = 0.025f;
    config.epochs = 50;
    config.min_count = 1;

    Word2Vec<float> w2v(config);

    std::cout << "Loading corpus..." << std::endl;
    w2v.load_corpus(sample_text);
    std::cout << "Vocabulary size: " << w2v.get_vocab_size() << " words" << std::endl;

    std::cout << "Preparing training data..." << std::endl;
    w2v.prepare_training_data();

    std::cout << "Training skip-gram model..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    w2v.train();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "\nTraining completed in " << duration.count() << " ms" << std::endl;

    // Test word vectors
    std::cout << "\n=== Word Similarity Tests ===" << std::endl;

    std::vector<std::string> test_words = {"fox", "dog", "learning", "the"};

    for (const auto& word : test_words) {
        try {
            std::cout << "\nMost similar to '" << word << "':" << std::endl;
            auto similar = w2v.most_similar(word, 5);
            for (const auto& [similar_word, score] : similar) {
                std::cout << "  " << similar_word << ": " << score << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << "  Error: " << e.what() << std::endl;
        }
    }

    // Save embeddings
    std::cout << "\nSaving embeddings..." << std::endl;
    w2v.save_embeddings("word2vec_embeddings");
    std::cout << "Saved to word2vec_embeddings.vocab and word2vec_embeddings.weights" << std::endl;
}

void demo_from_file(const std::string& filepath) {
    std::cout << "=== Training Word2Vec from File ===" << std::endl;
    std::cout << "File: " << filepath << std::endl;

    Word2Vec<float>::TrainingConfig config;
    config.embedding_dim = 100;
    config.window_size = 5;
    config.negative_samples = 5;
    config.learning_rate = 0.025f;
    config.epochs = 10;
    config.min_count = 2;

    Word2Vec<float> w2v(config);

    std::cout << "Loading corpus from file..." << std::endl;
    w2v.load_corpus_from_file(filepath);
    std::cout << "Total words: " << w2v.get_vocab_size() << std::endl;

    std::cout << "Preparing training data..." << std::endl;
    w2v.prepare_training_data();

    std::cout << "Training..." << std::endl;
    w2v.train();

    std::cout << "\nSaving embeddings..." << std::endl;
    w2v.save_embeddings("word2vec_file_embeddings");
    std::cout << "Saved to word2vec_file_embeddings.vocab and word2vec_file_embeddings.weights" << std::endl;
}

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --demo              Run demo with sample text (default)" << std::endl;
    std::cout << "  --file <path>       Train from a text file" << std::endl;
    std::cout << "  --help              Show this help message" << std::endl;
}

int main(int argc, char* argv[]) {
    try {
        if (argc < 2) {
            demo_with_sample_text();
        } else {
            std::string arg = argv[1];

            if (arg == "--help" || arg == "-h") {
                print_usage(argv[0]);
                return 0;
            } else if (arg == "--demo") {
                demo_with_sample_text();
            } else if (arg == "--file" && argc >= 3) {
                demo_from_file(argv[2]);
            } else {
                std::cerr << "Unknown option: " << arg << std::endl;
                print_usage(argv[0]);
                return 1;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
