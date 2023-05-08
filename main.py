from a4Zero import *
from a4One import *
from a4Two import *
from a4Three import *

plt.title("NN Training/Testing Graph")
plt.xlabel('Epoch')
plt.ylabel('Loss')

nnZero(plt)
nnOne(plt)
nnTwo(plt)
nnThree(plt)

plt.legend()
plt.show()
