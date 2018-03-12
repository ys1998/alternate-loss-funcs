#include <stdio.h>
#include <stdlib.h>

int f(int i, int j){
	return i*i+j*j+2*i*j;
}
int main(){
	for(int i=0; i<20000; ++i){
		for(int j=0;j<25; ++j){
			int temp = f(i,j);
		}
	}
	return 0;
}
