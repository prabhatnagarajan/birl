cdef extern from "birl.c":
	int fib(int n)

def main():
	print fib(5)