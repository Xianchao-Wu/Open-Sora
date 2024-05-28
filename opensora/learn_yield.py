def add(x):
    out = 0
    for i in range(x):
        out += i
        yield out
    return out

out = add(10)
#print(out)
for aout in out:
    print(aout)
