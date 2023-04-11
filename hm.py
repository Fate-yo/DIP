import os
import random

for i in range(1, 21):
    path = "data\\" + str(i)
    os.mkdir(path)
    for j in range(1, 21):
        with open(path + "\\" + str(j)+'.txt', 'a+') as f:
            ans = str(random.randint(100000, 999999))

            f.write(ans)
