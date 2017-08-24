cdef extern from "fib.c":
	int fib(int n)

def main():
	print fib(5)