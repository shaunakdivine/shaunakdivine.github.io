---
layout: default
title: Understanding Coin Flips, Probability, and Bayes' Rule – A Deeper Dive
date:   2024-09-12 2:06:00 -0500
categories: probability
permalink: /blog/blog1/

---

# Understanding Coin Flips, Probability, and Bayes' Rule – A Deeper Dive
### September 12, 2024

Recently, I received a thought-provoking technical question that explores probability and Bayes' Rule. It's a fun problem that seems complicated at first but can simplify down with some creative mental math. The question goes like this:

> You have 1000 coins, 1 of which has heads on both sides, and the other 999 are fair. You randomly choose a coin and toss it 10 times. Each time, it lands heads. What is the probability that you picked the unfair coin?

This problem is a classic application of **Bayes' Rule**, involving conditional probability. Here, the "condition" is that we flipped heads 10 times in a row. Intuitively, I first thought that it would be highly likely that the double-headed coin was picked, but let’s break down the math to really understand how it works.

## Solving the Original Problem with Bayes’ Rule

To find the probability that we picked the unfair coin given 10 heads, we’ll use Bayes' Rule:

$$ 
P(A|B) = \frac{P(B|A)P(A)}{P(B)} 
$$

Where:
- \( A \) is the event that we picked the unfair coin.
- \( B \) is the event that we observed 10 heads in a row.

So now let’s break this down step by step:

### 1. **Prior Probability \( P(A) \)**

The prior probability that we picked the unfair coin is the ratio of unfair coins to total coins:

$$ 
P(A) = \frac{1}{1000} 
$$

### 2. **Likelihood \( P(B|A) \)**

The likelihood is the probability of observing 10 heads given that we picked the unfair coin. Since this coin always lands heads, the probability is simply 1:

$$ 
P(B|A) = 1 
$$

### 3. **Complement Prior \($$ P(\neg A) $$\)**

The probability of not picking the unfair coin (picking a fair coin) is:

$$
P(\neg A) = \frac{999}{1000}
$$

### 4. **Complement Likelihood \($$ P(B|\neg A) $$\)**

Now, what’s the probability of getting 10 heads in a row given that we picked a fair coin? Each flip has a probability of \($$ \frac{1}{2} $$\) of landing heads, so for 10 heads in a row:

$$
P(B|\neg A) = \left(\frac{1}{2}\right)^{10} = \frac{1}{1024}
$$


Here's where the quick mental math comes into play. If you were like me and grew up playing 2048, we can quickly reason that \($$ 2^{10} $$\) is 1024. However, to simplify this problem, we can assume that:

$$
\left(\frac{1}{2}\right)^{10} \approx \frac{1}{1000}
$$

This assumption will help us later with simplifying the expression without the use of a calculator.


### 5. **Total Probability \( P(B) \)**

The total probability of observing 10 heads can be calculated using the law of total probability:

$$ P(B) = P(B|A)P(A) + P(B|\neg A)P(\neg A) $$

Substituting the values:

$$ 
P(B) = 1 \cdot \frac{1}{1000} + \frac{1}{1024} \cdot \frac{999}{1000}
$$

$$ 
P(B) = \frac{1}{1000} + \frac{999}{1024 \times 1000} 
$$

$$ 
P(B) \approx \frac{1}{1000} + \frac{0.975}{1000} = \frac{1.975}{1000} 
$$

However, if we apply our assumption from earlier, we can treat \( $$\frac{1}{1024} $$\) as approximately \( $$\frac{1}{1000} $$\) and \( $$\frac{999}{1000} $$\) as approximately 1. 

This gives us:

$$
P(B) \approx \frac{1}{1000} + \frac{1}{1000} = \frac{2}{1000}
$$


### 6. **Putting it All Together with Bayes' Rule**

Now, we can quickly calculate the posterior probability using Bayes’ Rule and our previous assumption:

$$ 
P(A|B) = \frac{P(B|A)P(A)}{P(B)} = \frac{1 \cdot \frac{1}{1000}}{\frac{2}{1000}} = \frac{1}{2} \approx 0.5 
$$

So, given 10 heads in a row, the probability that you picked the unfair coin simplifies down to approximately **50%**. I've also worked out the more precise answer below:

$$ 
P(A|B) = \frac{P(B|A)P(A)}{P(B)} = \frac{1 \cdot \frac{1}{1000}}{\frac{1.975}{1000}} = \frac{1}{1.975} \approx 0.506 
$$


## Solving the General Case

I remember solving this problem for the first time, and I was surprised to see the result of approximately 50%. While this was cool, I immediately wanted to explore the effect of changing some of the variables. At a core level, I knew the result depended on the ratio of unfair coins to total coins, so I first wanted to identify this relationship. I began by trying to find how to keep the result steady at 50% while altering some of the variables. Let's say the relationship between the number of unfair coins \( K \), the total number of coins \( N \), and the number of consecutive heads \( n \) is given by:

$$
P(U|H) = \frac{P(H|U) \cdot P(U)}{P(H|U) \cdot P(U) + P(H|F) \cdot P(F)}
$$

Where:
- \($$ P(U) = \frac{K}{N} $$\) is the prior probability of picking an unfair coin.

- \($$ P(F) = \frac{N - K}{N} $$\) is the prior probability of picking a fair coin.

- \($$ P(H \mid U) = 1 $$\), because an unfair coin (double-headed) always lands heads.

- \($$ P(H \mid F) = \left( \frac{1}{2} \right)^n $$\), because a fair coin has a 0.5 chance of landing heads each toss.


### Simplifying the Equation:

Plugging these probabilities into Bayes' theorem:

$$
P(U|H) = \frac{\frac{K}{N}}{\frac{K}{N} + \left( \frac{1}{2} \right)^n \cdot \frac{N - K}{N}} = \frac{K}{K + (N - K) \cdot \left( \frac{1}{2} \right)^n}
$$

$$
.5 = \frac{K}{K + (N - K) \cdot \left( \frac{1}{2} \right)^n}
$$

$$
K + (N - K) \cdot \left( \frac{1}{2} \right)^n = 2K
$$

$$
(N - K) \cdot \left( \frac{1}{2} \right)^n = K
$$

$$
\left( \frac{1}{2} \right)^n = \frac{K}{N - K}
$$

### Looking at the Ratio:

The ratio \($$ \frac{K}{N - K} $$\) must equal \($$ \left( \frac{1}{2} \right)^n $$\) to maintain a 50% posterior probability. This means:

- As \($$ n $$\) increases, \($$ \left( \frac{1}{2} \right)^n $$\) decreases exponentially.
- To compensate, the ratio \($$ \frac{K}{N - K} $$\) must also decrease exponentially.
- This requires increasing \($$ N $$\) exponentially while keeping \($$ K $$\) fixed or adjusting \($$ K $$\) accordingly.

Now we can see the general form of the equation and get a better understanding of how the variables affect the outcome. With this in mind, I then wanted to use Python to adjust the variables and get a direct representation of how the probability changes.

## Graphical Exploration


```python
import numpy as np
import matplotlib.pyplot as plt

def posterior_probability(K, N, n):
    numerator = K 
    denominator = K + (N - K) * (0.5) ** n
    return numerator / denominator

def plot_posterior_vs_N(K=1, n=10, N_max=5000):
    N_values = np.arange(K + 1, N_max)
    probabilities = [posterior_probability(K, N, n) for N in N_values]

    plt.figure(figsize=(10, 6))
    plt.plot(N_values, probabilities, label=f'K={K}, n={n}')
    plt.axhline(0.5, color='r', linestyle='--', label='P(U|H) = 0.5')
    plt.xlabel('Total Number of Coins (N)')
    plt.ylabel('Posterior Probability P(U|H)')
    plt.title('Posterior Probability vs. Total Number of Coins (N)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_posterior_vs_K(N=1000, n=10, K_max=100):
    K_values = np.arange(1, K_max)
    probabilities = [posterior_probability(K, N, n) for K in K_values]

    plt.figure(figsize=(10, 6))
    plt.plot(K_values, probabilities, label=f'N={N}, n={n}')
    plt.axhline(0.5, color='r', linestyle='--', label='P(U|H) = 0.5')
    plt.xlabel('Number of Unfair Coins (K)')
    plt.ylabel('Posterior Probability P(U|H)')
    plt.title('Posterior Probability vs. Number of Unfair Coins (K)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_posterior_vs_n(K=1, N=1000, n_max=20):
    n_values = np.arange(1, n_max)
    probabilities = [posterior_probability(K, N, n) for n in n_values]

    plt.figure(figsize=(10, 6))
    plt.plot(n_values, probabilities, label=f'K={K}, N={N}')
    plt.axhline(0.5, color='r', linestyle='--', label='P(U|H) = 0.5')
    plt.xlabel('Number of Consecutive Heads (n)')
    plt.ylabel('Posterior Probability P(U|H)')
    plt.title('Posterior Probability vs. Number of Consecutive Heads (n)')
    plt.legend()
    plt.grid(True)
    plt.show()


plot_posterior_vs_N(K=1, n=10, N_max=10000)
plot_posterior_vs_K(N=1000, n=10, K_max=200)
plot_posterior_vs_n(K=1, N=1000, n_max=25)

```


    
![png](/assets/coins_files/coins_1_0.png)
    



    
![png](/assets/coins_files/coins_1_1.png)
    



    
![png](/assets/coins_files/coins_1_2.png)
    


The first plot shows how the probability varies with the total number of coins when the amount of unfair coins and heads flipped are held constant. We can see how quickly the total number begins to impact the final probability drastically. When the total coins approaches 4,000, we can already see that the probability has dropped to approximately 20%. On the other end, we can see that if we cut the amount of coins in half to about 500, the probability increases to around 70%. 

This second graph explores how the posterior probability changes with the amount of unfair coins to choose from. This plot is interesting because it shows how drastically this factor changes the outcome. Even changing the percent of unfair coins from .1% to 2.5%, we can already see that the probability is close to 95%, approaching 100%. This goes to show the importance of this factor on the outcome.

The third graph shows how altering the number of consectutive heads affects the outcome. Initially, with a small number of heads, the posterior probability is very low, indicating a low likelihood of having picked the unfair coin. However, as the number of heads increases, the posterior probability grows exponentially and crosses the 50% threshold after about 10 consecutive heads. This factor also has a lot of variability, showing its impact on the outcome as well.

Finally, the code and plot below wrap up my discussion of this problem, as it shows a nice overview of my findings. The contour plot puts all the different probability levels on a single plot, showing how the number of consecutive heads and total coins intersect. 


```python
def plot_contour_N_n(K=1, N_max=2000, n_max=20):
    N_values = np.arange(K + 1, N_max)
    n_values = np.arange(1, n_max)
    N_grid, n_grid = np.meshgrid(N_values, n_values)
    probabilities = np.vectorize(posterior_probability)(K, N_grid, n_grid)

    plt.figure(figsize=(10, 6))
    cp = plt.contour(N_values, n_values, probabilities, levels=20)
    plt.clabel(cp, inline=True, fontsize=10)
    plt.xlabel('Total Number of Coins (N)')
    plt.ylabel('Number of Consecutive Heads (n)')
    plt.title('Contour Plot of P(U|H) over N and n')
    plt.show()

plot_contour_N_n(K=1, N_max=2000, n_max=20)

```


    
![png](/assets/coins_files/coins_3_0.png)
    

