static int fib(int n)
{
	int a = 0;
	int b = 1;
	int i;
	for (i = 0; i < n; i++)
	{
		int temp = b;
		b = b + a;
		a = temp;
	}
	return b;
}