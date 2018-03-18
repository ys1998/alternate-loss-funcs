#include "trie.hpp"
#include <pthread.h>

#define N_THREADS 28

float log_to_ln(float input) {
	return (input * LN_10);
}


Trie::Trie() {
	backoff = 0.0;
	log_prob = 0.0;
	character = "";
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

// Structure to store and pass data to threads
struct thread_data{
	Trie *ptr;
	int i, j;
	vector<int> context;
	int num_batches;
	int pointer;
	double* distro;
	int batch_size, timesteps, vocab_size;
    
    // Default constructor
	thread_data(){
		this->ptr = NULL;
		this->i = 0;
		this->j = 0;
		this->context = vector<int>(0);
		this->num_batches = 0;
		this->pointer = 0;
		this->distro = NULL;
		this->batch_size = 0;
		this->timesteps = 0;
		this->vocab_size = 0;
	}
    // Parameterized constructor
	thread_data(Trie *T, int i, int j, vector<int> cntxt, int nb, int ptr, double* d, int bs, int t, int vs){
		this->ptr = T;
		this->i = i;
		this->j = j;
		this->context = cntxt;
		this->num_batches = nb;
		this->pointer = ptr;
		this->distro = d;
		this->batch_size = bs;
		this->timesteps = t;
		this->vocab_size = vs;
	}
};

// Callable for every thread
void *gen_freq(void *thread_arg){
	// Typecast thread_arg from void* to ptr of structure
	thread_data *D = (thread_data *)thread_arg;
    
    // Find probability distribution for each i,j independently
	vector<int>::iterator context_start = D->context.begin();
	vector<int>::iterator context_end = D->context.end();
	int context_size = D->context.size();
	vector<int>::iterator last_word = context_end;
	advance(last_word, -1);
	float backoff = 0.0;
	for (int gram = 0; gram < context_size; gram++) {
		vector<int>::iterator it = context_start;
		advance(it, gram);
		Trie* current = D->ptr;
		while (it != context_end) {
			if (current->children.find(*it) == current->children.end()) {
				// Context is not found! Search for a smaller gram
				break;
			} else {
				// Context found, proceed in the trie.
				current = current->children[*it];
				if (it == last_word) {
					// Arrived at the end of the context, hunt for distribution
					for (map<int, Trie*>::iterator it2 = current->children.begin(); it2 != current->children.end(); it2++){
						if (D->distro[D->i*D->vocab_size*D->timesteps + D->j*D->vocab_size + it2->first] == 0.0) {
							// This indicates that this token has not been written
							// by a higher order gram.
							D->distro[D->i*D->vocab_size*D->timesteps + D->j*D->vocab_size + it2->first] = (it2->second)->log_prob + backoff;
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
	Trie* current = D->ptr;
	for (int k = 0; k < D->vocab_size; ++k) {
		if (D->distro[D->i*D->vocab_size*D->timesteps + D->j*D->vocab_size + k] == 0.0) {
			D->distro[D->i*D->vocab_size*D->timesteps + D->j*D->vocab_size + k] = (current->children[k])->log_prob + backoff;
		}
	}
}

void Trie::parallel_get_distro(vector< vector<int> > &context, int num_batches, int pointer, double* distro, int batch_size, int timesteps, int vocab_size){
    // Declare N_THREAD threads
	pthread_t thread[N_THREADS];
    // Variable to store exit status of threads
	void *status;
	int i=0, j=0, cntr=0;
	int total_iters = batch_size * timesteps;
	vector<thread_data> data(0);

	while(total_iters > 0){
		bool start = false;
		// Clear thread data
		data.resize(0);
		// Collect input data for atmost N_THREAD threads
		for (; i < batch_size; ++i){
			for (; j < timesteps; ++j){
				if(cntr >= N_THREADS){
					cntr = 0;
					start = true;
					break;
				}else{
					// Generate thread data here
					int index = i * timesteps * num_batches + pointer * timesteps + j;
					data.push_back(thread_data(this, i, j, context[index], num_batches, pointer, distro, batch_size, timesteps, vocab_size));
					cntr++;
					total_iters--;
				}
			}
			if(j == timesteps) j=0;
			if(start) break;
		}
		// Create as many threads as the number of thread_data instances
		// All threads are joinable by default
		for(int k=0; k<data.size(); ++k){
			pthread_create(&thread[k], NULL, gen_freq, (void *)&data[k]);
		}
		// Wait for all threads to join - change this if too much time required
		for(int k=0; k<data.size(); ++k){
			pthread_join(thread[k], &status);
		}
	}
}

void Trie::get_distro(vector< vector<int> > &context, int num_batches, int pointer, double* distro, int batch_size, int timesteps, int vocab_size){
	// Call parallelized code
    parallel_get_distro(context, num_batches, pointer, distro, batch_size, timesteps, vocab_size);
}