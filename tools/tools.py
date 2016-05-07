def debug(*args, **kwargs):
	if "level" not in kwargs or kwargs["level"] < 3:
		return
	print "DEBUG: ",
	for a in args:
		print a,
	print

warn = debug
