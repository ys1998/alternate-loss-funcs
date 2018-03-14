%module trie
%include <std_vector.i>
%include <std_list.i>
%include <std_unordered_map.i>
%include <std_string.i>

namespace std {
    %template(map_string_int) unordered_map<string, int>;
    %template(vector_float) vector<float>;
    %template(list_int) list<int>;
    %template(list_int_int) list< list<int> >;
}

%{
#define SWIG_FILE_WITH_INIT
#include "trie.hpp"
%}
%include "numpy.i"
%init %{
    import_array();
%}

class Trie {
public:
    std::unordered_map<int, Trie*> children;
    float backoff;
    float log_prob;
    std::string character;
    Trie();
    void load_arpa(std::string filename, std::unordered_map<string, int> &vocab);
    void get_distro(std::list< list<int> > &context, int num_batches, int vocab_size, int batch_size, int timesteps, int pointer, double* distro);
};