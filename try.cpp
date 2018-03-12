#include <iostream>
using namespace std;

long long int f(int i, int j){
	return i*i+j*j+2*i*j;
}

int main(){
	long long int I = 20000;
	long long int J = 25;
	for(int i=0; i<I; ++i){
		for(int j=0;j<J; ++j){
			long long int temp = f(i,j);
		}
	}
	return 0;
}
