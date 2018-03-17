#include "trie.hpp"

float log_to_ln(float input) {
	return (input * LN_10);
}


Trie::Trie() {
	backoff = 0.0;
	log_prob = 0.0;
	character = "";
}


void Trie::get_distro(vector< vector<int> > &context, int num_batches, int pointer, double* distro, int batch_size, int timesteps, int vocab_size){
	for(int i=0; i < batch_size; ++i){
		for(int j=0; j < timesteps; ++j){
			int index = i * timesteps * num_batches + pointer * timesteps + j;
			// cout<<context[index].size()<<" "<<context[index][0]<<endl;
			// context[index]
			// distro[i * batch_size * timesteps + j * timesteps + ]
			vector<int>::iterator context_start = context[index].begin();
			vector<int>::iterator context_end = context[index].end();
			int context_size = context[index].size();
			vector<int>::iterator last_word = context_end;
			advance(last_word, -1);
			float backoff = 0.0;
			for (int gram = 0; gram < context_size; gram++) {
				vector<int>::iterator it = context_start;
				advance(it, gram);
				Trie* current = this;
				while (it != context_end) {
					if (current->children.find(*it) == current->children.end()) {
						// Context is not found! Search for a smaller gram
						break;
					} else {
						// Context found, proceed in the trie.
						current = current->children[*it];
						if (it == last_word) {
							// Arrived at the end of the context, hunt for distribution
							for (map<int, Trie*>::iterator it2 = current->children.begin(); it2 != current->children.end(); ++it2){
								if (distro[i*batch_size*timesteps + j*timesteps + it2->first] == 0.0) {
									// This indicates that this token has not been written
									// by a higher order gram.
									distro[i*batch_size*timesteps + j*timesteps + it2->first] = (it2->second)->log_prob + backoff;
								}
							}
							// Update the backoff values
							backoff += current->backoff;
						}
					}
					it++;
				}
			}
			// Separately hunt for unigrams
			Trie* current = this;
			for (int k = 0; k < vocab_size; ++k) {
				if (distro[i*batch_size*timesteps + j*timesteps + k] == 0.0) {
					distro[i*batch_size*timesteps + j*timesteps + k] = (current->children[k])->log_prob + backoff;
				}
			}
		}
	}
}


void Trie::load_arpa(string filename, map<string, int> &vocab) {
	ifstream infile(filename.c_str());
	string line;
	vector<int> ngram_sizes;
	ngram_sizes.push_back(0);
	int stage = 0;
	int gram = 0;
	string token = "";
	while (getline(infile, line)) {
		istringstream iss(line);
		if (line.empty()) {
			stage += 1;
			continue;
		}
		switch (stage) {
			case 1:
				// These lines in ARPA indicate number of n-grams
				iss >> token;
				if (token == "\\data\\") {
					cout << "Reading # of ngrams..." << endl;
				} else if (token == "ngram") {
					iss >> token;
					int pos = token.find_first_of('=');
					int size = lexical_cast<int>(token.substr(pos + 1));
					ngram_sizes.push_back(size);
				}
				break;
			default:
				// These lines are the actual data
				iss >> token;
				if (token == "\\end\\") {
					break;
				} else if (token[0] == '\\') {
					int pos = token.find_first_of('-');
					gram = lexical_cast<int>(token.substr(1, pos - 1));
					cout << "Loading " << ngram_sizes[gram] << " " <<  gram << "-grams" << endl;
					break;
				}
				float log_prob = log_to_ln(lexical_cast<float>(token));
				Trie* current = this;
				// Iterate over the tokens
				bool created = false;
				for(int i = 0; i < gram; i++) {
					iss >> token;
					int vocab_id = vocab[token];
					if (current->children.find(vocab_id) == current->children.end()) {
						Trie* t = new Trie();
						current->children.insert(pair<int, Trie*>(vocab_id, t));
						created = true;
					}
					current = current->children[vocab_id];
				}
				if (created == false) {
					// Sanity check, every unique n-gram must create a new Trie
					cout << "Error" << endl;
				}
				current->log_prob = log_prob;
				current->character = token;
				// Some tokens don't have backoff weights
				if (iss.eof()) {
					current->backoff = 0.0;
				} else {
					iss >> token;
					current->backoff = log_to_ln(lexical_cast<float>(token));
				}
		}
	}
	infile.close();
}