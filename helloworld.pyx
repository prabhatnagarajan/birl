cdef extern from "fib.c":
	int fib(int n)

print fib(5)