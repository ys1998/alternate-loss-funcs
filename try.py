def f(i,j):
    return i*i+j*j+2*i*j

def main():
    for i in range(20000):
        for j in range(25):
            temp = f(i,j)


if __name__=='__main__':
    main()
