{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1.16.4\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "\n",
    "print(np.__version__)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0 1 2 3 4 5 6 7 8 9]\n"
     ]
    }
   ],
   "source": [
    "arr = np.arange(10)\n",
    "\n",
    "print(arr)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[False False False]\n [False False False]\n [False False False]]\n"
     ]
    }
   ],
   "source": [
    "print(np.full((3, 3), False, dtype=bool))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1 3 5 7 9]\n"
     ]
    }
   ],
   "source": [
    "print(arr[arr % 2 == 1])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[ 0 -1  2 -1  4 -1  6 -1  8 -1]\n"
     ]
    }
   ],
   "source": [
    "arr[arr % 2 == 1] = -1\n",
    "\n",
    "print(arr)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0 1 2 3 4]\n [5 6 7 8 9]]\n"
     ]
    }
   ],
   "source": [
    "arr = np.arange(10)\n",
    "\n",
    "print(arr.reshape(2, -1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0 1 2 3 4]\n [5 6 7 8 9]\n [1 1 1 1 1]\n [1 1 1 1 1]]\n\n [[0 1 2 3 4]\n [5 6 7 8 9]\n [1 1 1 1 1]\n [1 1 1 1 1]]\n"
     ]
    }
   ],
   "source": [
    "a = arr.reshape(2, -1)\n",
    "b = np.repeat(1, 10).reshape(2, -1)\n",
    "\n",
    "print(np.concatenate([a, b], axis=0))\n",
    "\n",
    "print('\\n', np.vstack([a, b]))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 2, 3, 2, 3, 4, 3, 4, 5, 6])"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "print(np.hstack([a, b]))\n",
    "print('\\n')\n",
    "print(np.hstack([a,b]))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 2, 3, 2, 3, 4, 3, 4, 5, 6])"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "a = np.array([1,2,3,2,3,4,3,4,5,6])\n",
    "\n",
    "b = np.array([7,2,10,2,7,4,9,4,9,8])\n",
    "\n",
    "a"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[2 4]\n"
     ]
    }
   ],
   "source": [
    "print(np.intersect1d(a,b))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1 3 5 6]\n"
     ]
    }
   ],
   "source": [
    "# From 'a' remove all of 'b'\n",
    "\n",
    "print(np.setdiff1d(a,b))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(array([1, 3, 5, 7], dtype=int64),)\n"
     ]
    }
   ],
   "source": [
    "print(np.where(a == b))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0 1 2 3 4]\n [5 6 7 8 9]]\n\n\n[[1 1 1 1 1]\n [1 1 1 1 1]]\n"
     ]
    }
   ],
   "source": [
    "print(a)\n",
    "print('\\n')\n",
    "print(b)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(array([5, 7, 8, 9], dtype=int64),)\n[5 6]\n"
     ]
    }
   ],
   "source": [
    "index = np.where((a >= 4) & (a <= 6))\n",
    "\n",
    "print(index)\n",
    "\n",
    "print(a[(a >= 5) & (a <= 10)])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0 1 2]\n [3 4 5]\n [6 7 8]]\n\n\n[[1 0 2]\n [4 3 5]\n [7 6 8]]\n"
     ]
    }
   ],
   "source": [
    "arr = np.arange(9).reshape(3,3)\n",
    "\n",
    "print(arr)\n",
    "print('\\n')\n",
    "print(arr[:, [1,0,2]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[6, 7, 8],\n       [3, 4, 5],\n       [0, 1, 2]])"
      ]
     },
     "execution_count": 68,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr = np.arange(9).reshape(3,3) #rows reverting\n",
    "\n",
    "arr[::-1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[5.70984603 7.56226873 5.51651151]\n [7.57376052 6.53382225 5.61077057]\n [7.51302643 6.55047676 5.4517255 ]\n [9.53163278 5.96964337 6.64131797]\n [8.37449981 9.262355   8.81312326]]\n"
     ]
    }
   ],
   "source": [
    "rand_arr = np.random.randint(low=5, high=10, size=(5,3)) + np.random.random((5,3))\n",
    "\n",
    "rand_arr = np.random.uniform(5,10, size=(5,3))\n",
    "\n",
    "print(rand_arr)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0.000543, 0.000278, 0.000425],\n       [0.000845, 0.000005, 0.000122],\n       [0.000671, 0.000826, 0.000137]])"
      ]
     },
     "execution_count": 77,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Reset printoptions to default\n",
    "np.set_printoptions(suppress=False)\n",
    "\n",
    "# Create the random array\n",
    "np.random.seed(100)\n",
    "rand_arr = np.random.random([3, 3]) / 1e3\n",
    "rand_arr\n",
    "\n",
    "np.set_printoptions(suppress=True, precision=6)  # precision is optional\n",
    "rand_arr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 0,  1,  2, ..., 12, 13, 14])"
      ]
     },
     "execution_count": 86,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.set_printoptions(threshold=10)\n",
    "a = np.arange(15)\n",
    "a"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 108,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 0,  1,  2, ..., 12, 13, 14])"
      ]
     },
     "execution_count": 108,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import sys\n",
    "\n",
    "np.set_printoptions(threshold=10)\n",
    "\n",
    "a"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[b'5.1', b'3.5', b'1.4', b'0.2', b'Iris-setosa'],\n       [b'4.9', b'3.0', b'1.4', b'0.2', b'Iris-setosa'],\n       [b'4.7', b'3.2', b'1.3', b'0.2', b'Iris-setosa']], dtype=object)"
      ]
     },
     "execution_count": 90,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'\n",
    "iris = np.genfromtxt(url, delimiter=',', dtype='object')\n",
    "names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')\n",
    "\n",
    "iris[:3]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([4.6, 5.5])"
      ]
     },
     "execution_count": 104,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])\n",
    "\n",
    "Smax, Smin = sepallength.max(), sepallength.min()\n",
    "\n",
    "print(Smax)\n",
    "\n",
    "sepallength"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 104,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([4.6, 5.5])"
      ]
     },
     "execution_count": 104,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.percentile(sepallength, q=[5, 95])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 109,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[b'5.1' b'3.5' b'1.4' b'0.2' b'Iris-setosa']\n [b'4.9' b'3.0' b'1.4' b'0.2' b'Iris-setosa']\n [b'4.7' b'3.2' b'1.3' b'0.2' b'Iris-setosa']\n ...\n [b'6.5' b'3.0' b'5.2' b'2.0' nan]\n [b'6.2' b'3.4' b'5.4' b'2.3' b'Iris-virginica']\n [b'5.9' b'3.0' b'5.1' b'1.8' b'Iris-virginica']]\n"
     ]
    }
   ],
   "source": [
    "url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'\n",
    "iris_2d = np.genfromtxt(url, delimiter=',', dtype='object')\n",
    "\n",
    "i, j = np.where(iris_2d)\n",
    "\n",
    "np.random.seed(100)\n",
    "iris_2d[np.random.choice((i), 20), np.random.choice((j), 20)] = np.nan\n",
    "\n",
    "print(iris_2d)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 124,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Number of missing values: \n 8\nPosition of missing values: \n (array([ 27,  55,  75,  88, 100, 111, 135, 142], dtype=int64),)\n"
     ]
    }
   ],
   "source": [
    "url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'\n",
    "iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0, 1, 2, 3])\n",
    "\n",
    "iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan\n",
    "\n",
    "print(\"Number of missing values: \\n\", np.isnan(iris_2d[:, 0]).sum())\n",
    "\n",
    "print(\"Position of missing values: \\n\", np.where(np.isnan(iris_2d[:, 0])))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 126,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "E:\\Programming\\Anaconda\\lib\\site-packages\\ipykernel_launcher.py:1: RuntimeWarning: invalid value encountered in greater\n  \"\"\"Entry point for launching an IPython kernel.\nE:\\Programming\\Anaconda\\lib\\site-packages\\ipykernel_launcher.py:1: RuntimeWarning: invalid value encountered in less\n  \"\"\"Entry point for launching an IPython kernel.\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array([[4.8, 3.4, 1.6, 0.2],\n       [4.8, 3.4, 1.9, 0.2],\n       [4.7, 3.2, 1.6, nan],\n       [4.8, 3.1, 1.6, 0.2],\n       [4.9, 2.4, 3.3, 1. ],\n       [4.9, 2.5, 4.5, 1.7]])"
      ]
     },
     "execution_count": 126,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "condition = (iris_2d[:, 2] > 1.5) & (iris_2d[:, 0] < 5.0)\n",
    "iris_2d[condition]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 132,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[5.1, 3.5, 1.4, 0.2],\n       [4.7, 3.2, 1.3, 0.2],\n       [4.6, 3.1, 1.5, 0.2],\n       [5. , 3.6, 1.4, 0.2],\n       [5.4, 3.9, 1.7, 0.4]])"
      ]
     },
     "execution_count": 132,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'\n",
    "iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])\n",
    "iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan\n",
    "\n",
    "any_nan_in_row = np.array([~np.any(np.isnan(row)) for row in iris_2d])\n",
    "iris_2d[any_nan_in_row][:5]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 139,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.8717541573048712\n"
     ]
    }
   ],
   "source": [
    "# Input\n",
    "url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'\n",
    "iris = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0, 1, 2, 3])\n",
    "\n",
    "# Solution 1\n",
    "print(np.corrcoef(iris[:, 0], iris[:, 2])[0, 1])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 147,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "True"
      ]
     },
     "execution_count": 147,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'\n",
    "iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])\n",
    "iris_2d[0][1] = np.nan\n",
    "np.isnan(iris_2d).any()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 169,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[5.1, 3.5, 1.4, 0.2],\n       [4.9, 1. , 1.4, 0.2],\n       [1. , 1. , 1.3, 1. ],\n       [1. , 1. , 1.5, 1. ]])"
      ]
     },
     "execution_count": 169,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Input\n",
    "url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'\n",
    "iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0, 1, 2, 3])\n",
    "iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan\n",
    "\n",
    "# Solution\n",
    "iris_2d[np.isnan(iris_2d)] = 0\n",
    "iris_2d[:4]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 181,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([b'Iris-setosa', b'Iris-setosa', b'Iris-setosa', ...,\n       b'Iris-virginica', b'Iris-virginica', b'Iris-virginica'],\n      dtype='|S15')"
      ]
     },
     "execution_count": 181,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'\n",
    "iris = np.genfromtxt(url, delimiter=',', dtype='object')\n",
    "names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')\n",
    "\n",
    "# Solution\n",
    "# Extract the species column as an array\n",
    "species = np.array([row.tolist()[4] for row in iris])\n",
    "\n",
    "species"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 184,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['small',\n 'small',\n 'small',\n 'small',\n 'small',\n 'small',\n 'small',\n 'small',\n 'small',\n 'small',\n 'small',\n 'small',\n 'small',\n 'small',\n 'small',\n 'small',\n 'small',\n 'small',\n 'small',\n 'small',\n 'small',\n 'small',\n 'small',\n 'small',\n 'small',\n 'small',\n 'small',\n 'small',\n 'small',\n 'small',\n 'small',\n 'small',\n 'small',\n 'small',\n 'small',\n 'small',\n 'small',\n 'small',\n 'small',\n 'small',\n 'small',\n 'small',\n 'small',\n 'small',\n 'small',\n 'small',\n 'small',\n 'small',\n 'small',\n 'small',\n 'medium',\n 'medium',\n 'medium',\n 'medium',\n 'medium',\n 'medium',\n 'medium',\n 'medium',\n 'medium',\n 'medium',\n 'medium',\n 'medium',\n 'medium',\n 'medium',\n 'medium',\n 'medium',\n 'medium',\n 'medium',\n 'medium',\n 'medium',\n 'medium',\n 'medium',\n 'medium',\n 'medium',\n 'medium',\n 'medium',\n 'medium',\n 'large',\n 'medium',\n 'medium',\n 'medium',\n 'medium',\n 'medium',\n 'large',\n 'medium',\n 'medium',\n 'medium',\n 'medium',\n 'medium',\n 'medium',\n 'medium',\n 'medium',\n 'medium',\n 'medium',\n 'medium',\n 'medium',\n 'medium',\n 'medium',\n 'medium',\n 'medium',\n 'large',\n 'large',\n 'large',\n 'large',\n 'large',\n 'large',\n 'medium',\n 'large',\n 'large',\n 'large',\n 'large',\n 'large',\n 'large',\n 'large',\n 'large',\n 'large',\n 'large',\n 'large',\n 'large',\n 'large',\n 'large',\n 'medium',\n 'large',\n 'medium',\n 'large',\n 'large',\n 'medium',\n 'medium',\n 'large',\n 'large',\n 'large',\n 'large',\n 'large',\n 'large',\n 'large',\n 'large',\n 'large',\n 'large',\n 'medium',\n 'large',\n 'large',\n 'large',\n 'large',\n 'large',\n 'large',\n 'large',\n 'large',\n 'large',\n 'large',\n 'large']"
      ]
     },
     "execution_count": 184,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Input\n",
    "url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'\n",
    "iris = np.genfromtxt(url, delimiter=',', dtype='object')\n",
    "names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')\n",
    "\n",
    "# Bin petallength \n",
    "petal_length_bin = np.digitize(iris[:, 2].astype('float'), [0, 3, 5, 10])\n",
    "\n",
    "# Map it to respective category\n",
    "label_map = {1: 'small', 2: 'medium', 3: 'large', 4: np.nan}\n",
    "petal_length_cat = [label_map[x] for x in petal_length_bin]\n",
    "\n",
    "# View\n",
    "petal_length_cat[:4]\n",
    "\n",
    "petal_length_cat"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 196,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[ 38.13265162927291   35.200498485922445  30.0723720777127   ...\n 230.0693019978925   217.373078887185    185.9100284614832  ]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array([[b'5.1', b'3.5', b'1.4', b'0.2', b'Iris-setosa',\n        38.13265162927291],\n       [b'4.9', b'3.0', b'1.4', b'0.2', b'Iris-setosa',\n        35.200498485922445],\n       [b'4.7', b'3.2', b'1.3', b'0.2', b'Iris-setosa', 30.0723720777127],\n       ...,\n       [b'6.5', b'3.0', b'5.2', b'2.0', b'Iris-virginica',\n        230.0693019978925],\n       [b'6.2', b'3.4', b'5.4', b'2.3', b'Iris-virginica',\n        217.373078887185],\n       [b'5.9', b'3.0', b'5.1', b'1.8', b'Iris-virginica',\n        185.9100284614832]], dtype=object)"
      ]
     },
     "execution_count": 196,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Input\n",
    "url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'\n",
    "iris_2d = np.genfromtxt(url, delimiter=',', dtype='object')\n",
    "\n",
    "# Solution\n",
    "# Compute volume\n",
    "sepallength = iris_2d[:, 0].astype('float')\n",
    "petallength = iris_2d[:, 2].astype('float')\n",
    "volume = (np.pi * petallength * (sepallength**2))/3\n",
    "\n",
    "# Introduce new dimension to match iris_2d's\n",
    "volume = volume[:, np.newaxis]\n",
    "\n",
    "# Add the new column\n",
    "out = np.hstack([iris_2d, volume])\n",
    "\n",
    "# View\n",
    "out"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 197,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1.7"
      ]
     },
     "execution_count": 197,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'\n",
    "iris = np.genfromtxt(url, delimiter=',', dtype='object')\n",
    "\n",
    "# Solution\n",
    "# Get the species and petal length columns\n",
    "petal_len_setosa = iris[iris[:, 4] == b'Iris-setosa', [2]].astype('float')\n",
    "\n",
    "# Get the second last value\n",
    "np.unique(np.sort(petal_len_setosa))[-2]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 198,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[b'4.3' b'3.0' b'1.1' b'0.1' b'Iris-setosa']\n [b'4.4' b'3.2' b'1.3' b'0.2' b'Iris-setosa']\n [b'4.4' b'3.0' b'1.3' b'0.2' b'Iris-setosa']\n ...\n [b'4.9' b'2.5' b'4.5' b'1.7' b'Iris-virginica']\n [b'4.9' b'3.1' b'1.5' b'0.1' b'Iris-setosa']\n [b'4.9' b'3.1' b'1.5' b'0.1' b'Iris-setosa']]\n"
     ]
    }
   ],
   "source": [
    "url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'\n",
    "iris = np.genfromtxt(url, delimiter=',', dtype='object')\n",
    "names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')\n",
    "\n",
    "# Sort by column position 0: SepalLength\n",
    "print(iris[iris[:,0].argsort()][:20])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 204,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "b'1.5'\n"
     ]
    }
   ],
   "source": [
    "url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'\n",
    "iris = np.genfromtxt(url, delimiter=',', dtype='object')\n",
    "names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')\n",
    "\n",
    "# Solution:\n",
    "vals, counts = np.unique(iris[:, 2], return_counts=True)\n",
    "\n",
    "print(vals[np.argmax(counts)])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 205,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([50], dtype=int64)"
      ]
     },
     "execution_count": 205,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Input:\n",
    "url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'\n",
    "iris = np.genfromtxt(url, delimiter=',', dtype='object')\n",
    "\n",
    "# Solution: (edit: changed argmax to argwhere. Thanks Rong!)\n",
    "np.argwhere(iris[:, 3].astype(float) > 1.0)[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 206,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[27.63 14.64 21.8  ... 10.   30.   14.43]\n"
     ]
    }
   ],
   "source": [
    "# Input\n",
    "np.set_printoptions(precision=2)\n",
    "np.random.seed(100)\n",
    "a = np.random.uniform(1,50, 20)\n",
    "\n",
    "# Solution 1: Using np.clip\n",
    "np.clip(a, a_min=10, a_max=30)\n",
    "\n",
    "# Solution 2: Using np.where\n",
    "print(np.where(a < 10, 10, np.where(a > 30, 30, a)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 209,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[27.63 14.64 21.8  ... 10.   30.   14.43]\n"
     ]
    }
   ],
   "source": [
    "# Input\n",
    "np.set_printoptions(precision=2)\n",
    "np.random.seed(100)\n",
    "a = np.random.uniform(1,50, 20)\n",
    "\n",
    "# Solution 1: Using np.clip\n",
    "np.clip(a, a_min=10, a_max=30)\n",
    "\n",
    "# Solution 2: Using np.where\n",
    "print(np.where(a < 10, 10, np.where(a > 30, 30, a)))\n",
    "#> [ 27.63  14.64  21.8   30.    10.    10.    30.    30.    10.    29.18  30.\n",
    "#>   11.25  10.08  10.    11.77  30.    30.    10.    30.    14.43]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 212,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[ 4 13  5 ...  3 10 15]\n"
     ]
    }
   ],
   "source": [
    "np.random.seed(100)\n",
    "a = np.random.uniform(1,50, 20)\n",
    "\n",
    "# Solution:\n",
    "print(a.argsort())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 213,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "array_of_arrays:  [array([0, 1, 2]) array([3, 4, 5, 6]) array([7, 8, 9])]\n[0 1 2 3 4 5 6 7 8 9]\n"
     ]
    }
   ],
   "source": [
    "# Input:\n",
    "arr1 = np.arange(3)\n",
    "arr2 = np.arange(3,7)\n",
    "arr3 = np.arange(7,10)\n",
    "\n",
    "array_of_arrays = np.array([arr1, arr2, arr3])\n",
    "print('array_of_arrays: ', array_of_arrays)\n",
    "\n",
    "# Solution 1\n",
    "arr_2d = np.array([a for arr in array_of_arrays for a in arr])\n",
    "\n",
    "# Solution 2:\n",
    "arr_2d = np.concatenate(array_of_arrays)\n",
    "print(arr_2d)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 216,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5]\n"
     ]
    }
   ],
   "source": [
    "# Input:\n",
    "url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'\n",
    "species = np.genfromtxt(url, delimiter=',', dtype='str', usecols=4)\n",
    "np.random.seed(100)\n",
    "species_small = np.sort(np.random.choice(species, size=20))\n",
    "species_small\n",
    "\n",
    "print([i for val in np.unique(species_small) for i, grp in enumerate(species_small[species_small==val])])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 218,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]\n"
     ]
    }
   ],
   "source": [
    "url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'\n",
    "species = np.genfromtxt(url, delimiter=',', dtype='str', usecols=4)\n",
    "np.random.seed(100)\n",
    "species_small = np.sort(np.random.choice(species, size=20))\n",
    "species_small\n",
    "\n",
    "# Solution:\n",
    "output = [np.argwhere(np.unique(species_small) == s).tolist()[0][0] for val in np.unique(species_small) for s in species_small[species_small==val]]\n",
    "\n",
    "# Solution: For Loop version\n",
    "output = []\n",
    "uniqs = np.unique(species_small)\n",
    "\n",
    "for val in uniqs:  # uniq values in group\n",
    "    for s in species_small[species_small==val]:  # each element in group\n",
    "        groupid = np.argwhere(uniqs == s).tolist()[0][0]  # groupid\n",
    "        output.append(groupid)\n",
    "\n",
    "print(output)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 220,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Array:  [ 9  4 15  0 17 16 17  8  9  0]\n[4 2 6 0 8 7 9 3 5 1]\nArray:  [ 9  4 15  0 17 16 17  8  9  0]\n"
     ]
    }
   ],
   "source": [
    "np.random.seed(10)\n",
    "a = np.random.randint(20, size=10)\n",
    "print('Array: ', a)\n",
    "\n",
    "# Solution\n",
    "print(a.argsort().argsort())\n",
    "print('Array: ', a)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 221,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 9  4 15  0 17]\n [16 17  8  9  0]]\n[[4 2 6 0 8]\n [7 9 3 5 1]]\n"
     ]
    }
   ],
   "source": [
    "# Input:\n",
    "np.random.seed(10)\n",
    "a = np.random.randint(20, size=[2,5])\n",
    "print(a)\n",
    "\n",
    "# Solution\n",
    "print(a.ravel().argsort().argsort().reshape(a.shape))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 222,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([9, 8, 6, 3, 9])"
      ]
     },
     "execution_count": 222,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Input\n",
    "np.random.seed(100)\n",
    "a = np.random.randint(1,10, [5,3])\n",
    "a\n",
    "\n",
    "# Solution 1\n",
    "np.amax(a, axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 223,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.44, 0.12, 0.5 , 1.  , 0.11])"
      ]
     },
     "execution_count": 223,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Input\n",
    "np.random.seed(100)\n",
    "a = np.random.randint(1,10, [5,3])\n",
    "a\n",
    "\n",
    "# Solution\n",
    "np.apply_along_axis(lambda x: np.min(x)/np.max(x), arr=a, axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 227,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[False  True False  True False False  True  True  True  True]\n10\n"
     ]
    }
   ],
   "source": [
    "# Input\n",
    "np.random.seed(100)\n",
    "a = np.random.randint(0, 5, 10)\n",
    "\n",
    "## Solution\n",
    "# There is no direct function to do this as of 1.13.3\n",
    "\n",
    "# Create an all True array\n",
    "out = np.full(a.shape[0], True)\n",
    "\n",
    "# Find the index positions of unique elements\n",
    "unique_positions = np.unique(a, return_index=True)[1]\n",
    "\n",
    "# Mark those positions as False\n",
    "out[unique_positions] = False\n",
    "\n",
    "print(out)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 237,
   "metadata": {},
   "outputs": [
    {
     "ename": "ImportError",
     "evalue": "DLL load failed: The specified module could not be found.",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mImportError\u001b[0m                               Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-237-39f81c586f66>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[1;32mfrom\u001b[0m \u001b[0mio\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mBytesIO\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 2\u001b[1;33m \u001b[1;32mfrom\u001b[0m \u001b[0mPIL\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mImage\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      3\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mPIL\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mrequests\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      5\u001b[0m \u001b[1;31m# Import image from URL\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mE:\\Programming\\Anaconda\\lib\\site-packages\\PIL\\Image.py\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m     92\u001b[0m     \u001b[1;31m# Also note that Image.core is not a publicly documented interface,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     93\u001b[0m     \u001b[1;31m# and should be considered private and subject to change.\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 94\u001b[1;33m     \u001b[1;32mfrom\u001b[0m \u001b[1;33m.\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0m_imaging\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0mcore\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     95\u001b[0m     \u001b[1;32mif\u001b[0m \u001b[0m__version__\u001b[0m \u001b[1;33m!=\u001b[0m \u001b[0mgetattr\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mcore\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m'PILLOW_VERSION'\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     96\u001b[0m         raise ImportError(\"The _imaging extension was built for another \"\n",
      "\u001b[1;31mImportError\u001b[0m: DLL load failed: The specified module could not be found."
     ],
     "output_type": "error"
    }
   ],
   "source": [
    "from io import BytesIO\n",
    "from PIL import Image\n",
    "import PIL, requests\n",
    "\n",
    "# Import image from URL\n",
    "URL = 'https://upload.wikimedia.org/wikipedia/commons/8/8b/Denali_Mt_McKinley.jpg'\n",
    "response = requests.get(URL)\n",
    "\n",
    "# Read it as Image\n",
    "I = Image.open(BytesIO(response.content))\n",
    "\n",
    "# Optionally resize\n",
    "I = I.resize([150,150])\n",
    "\n",
    "# Convert to numpy array\n",
    "arr = np.asarray(I)\n",
    "\n",
    "# Optionaly Convert it back to an image and show\n",
    "im = PIL.Image.fromarray(np.uint8(arr))\n",
    "Image.Image.show(im)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 240,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1., 2., 3., 5., 6., 7.])"
      ]
     },
     "execution_count": 240,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a = np.array([1,2,3,np.nan,5,6,7,np.nan])\n",
    "\n",
    "a[~np.isnan(a)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 241,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "6.708203932499369"
      ]
     },
     "execution_count": 241,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Input\n",
    "a = np.array([1,2,3,4,5])\n",
    "b = np.array([4,5,6,7,8])\n",
    "\n",
    "# Solution\n",
    "dist = np.linalg.norm(a-b)\n",
    "dist"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 242,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([2, 5], dtype=int64)"
      ]
     },
     "execution_count": 242,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a = np.array([1, 3, 7, 1, 2, 6, 0, 1])\n",
    "doublediff = np.diff(np.sign(np.diff(a)))\n",
    "peak_locations = np.where(doublediff == -2)[0] + 1\n",
    "peak_locations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 243,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[2 2 2]\n [2 2 2]\n [2 2 2]]\n"
     ]
    }
   ],
   "source": [
    "# Input\n",
    "a_2d = np.array([[3,3,3],[4,4,4],[5,5,5]])\n",
    "b_1d = np.array([1,2,3])\n",
    "\n",
    "# Solution\n",
    "print(a_2d - b_1d[:,None])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 244,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "8"
      ]
     },
     "execution_count": 244,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x = np.array([1, 2, 1, 1, 3, 4, 3, 1, 1, 2, 1, 1, 2])\n",
    "n = 5\n",
    "\n",
    "# Solution 1: List comprehension\n",
    "[i for i, v in enumerate(x) if v == 1][n-1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 245,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "datetime.datetime(2018, 2, 25, 22, 10, 10)"
      ]
     },
     "execution_count": 245,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Input: a numpy datetime64 object\n",
    "dt64 = np.datetime64('2018-02-25 22:10:10')\n",
    "\n",
    "# Solution\n",
    "from datetime import datetime\n",
    "dt64.tolist()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 246,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "array:  [8 8 3 7 7 0 4 2 5 2]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array([6.33, 6.  , 5.67, 4.67, 3.67, 2.  , 3.67, 3.  ])"
      ]
     },
     "execution_count": 246,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.random.seed(100)\n",
    "Z = np.random.randint(10, size=10)\n",
    "\n",
    "# Solution\n",
    "# Source: https://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy\n",
    "def moving_average(a, n=3) :\n",
    "    ret = np.cumsum(a, dtype=float)\n",
    "    ret[n:] = ret[n:] - ret[:-n]\n",
    "    return ret[n - 1:] / n\n",
    "\n",
    "np.random.seed(100)\n",
    "Z = np.random.randint(10, size=10)\n",
    "print('array: ', Z)\n",
    "# Method 1\n",
    "moving_average(Z, n=3).round(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 247,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 5,  8, 11, 14, 17, 20, 23, 26, 29, 32])"
      ]
     },
     "execution_count": 247,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "length = 10\n",
    "start = 5\n",
    "step = 3\n",
    "\n",
    "def seq(start, length, step):\n",
    "    end = start + (step*length)\n",
    "    return np.arange(start, end, step)\n",
    "\n",
    "seq(start, length, step)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 248,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 0  1  2  3]\n [ 2  3  4  5]\n [ 4  5  6  7]\n [ 6  7  8  9]\n [ 8  9 10 11]\n [10 11 12 13]]\n"
     ]
    }
   ],
   "source": [
    "arr = np.arange(15)\n",
    "\n",
    "def gen_strides(a, stride_len=5, window_len=5):\n",
    "    n_strides = ((a.size-window_len)//stride_len) + 1\n",
    "    # return np.array([a[s:(s+window_len)] for s in np.arange(0, a.size, stride_len)[:n_strides]])\n",
    "    return np.array([a[s:(s+window_len)] for s in np.arange(0, n_strides*stride_len, stride_len)])\n",
    "\n",
    "print(gen_strides(np.arange(15), stride_len=2, window_len=4))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 0
}
