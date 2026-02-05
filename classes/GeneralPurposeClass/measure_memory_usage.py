import sys

def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

def print_memory_usage(locals): #e.g. locals = locals(), but must be fetched at runtime
    ''' Prints memory usage of the current process '''
    total_size = 0
    print("---------- Memory usage (begin) ----------")
    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in list(locals.items())), key= lambda x: -x[1])[:10]:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
        total_size += size
    print("---------- Total size used: %s ----------" % sizeof_fmt(total_size))
    print("---------- Memory usage (end) ----------")