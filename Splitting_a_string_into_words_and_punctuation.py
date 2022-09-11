import re

s = "Hello, I'm a string! 40$ $40\\[]-"
s2 = ' '.join(re.findall(r"[\w']+|[!#$%&'()*+,\-./:;<=>?@\]\[\\^_`{|}~]", s))

print(s2)
