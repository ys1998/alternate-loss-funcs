// #include <map>
#include <vector>
#include <list>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <boost/lexical_cast.hpp>
// My addition (works for C++11)
#include <unordered_map>

using std::advance;
using boost::lexical_cast;
// using std::map;
using std::pair;
using std::vector;
using std::list;
using std::string;
using std::endl;
using std::ifstream;
using std::istringstream;
using std::cout;
using std::unordered_map;

#define LN_10 2.30258509299
#define LIMIT -INFINITY

class Trie {
public:
    unordered_map<int, Trie*> children;
    float backoff;
    float log_prob;
    string character;
    Trie();
    void load_arpa(string filename, unordered_map<string, int> &vocab);
    void get_distro(list<int> &context, double* distro, int distro_size);
};

