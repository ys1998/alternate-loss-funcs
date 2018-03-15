%module trie
%include <std_vector.i>
%include <std_map.i>
%include <std_string.i>

namespace std {
    %template(map_string_int) map<string, int>;
    %template(vector_float) vector<float>;
    %template(vector_int) vector<int>;
    %template(vector_vector_int) vector<vector<int> >;
}


%{
#define SWIG_FILE_WITH_INIT
#include "trie.hpp"
%}
%include "numpy.i"
%init %{
    import_array();
%}

%apply (double* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) {(double* distro, int batch_size, int timesteps, int vocab_size)};

class Trie {
public:
    std::map<int, Trie*> children;
    float backoff;
    float log_prob;
    std::string character;
    Trie();
    void load_arpa(std::string filename, std::map<string, int> &vocab);
    void get_distro(std::vector<std::vector<int> > &context, int num_batches, int pointer, double* distro, int batch_size, int timesteps, int vocab_size);
};
