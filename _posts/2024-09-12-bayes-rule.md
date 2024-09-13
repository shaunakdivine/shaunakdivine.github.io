---
layout: default
title: Understanding Coin Flips, Probability, and Bayes' Rule – A Deeper Dive
date:   2024-09-12 2:06:00 -0500
categories: probability
permalink: /blog/blog1/

---

# Understanding Coin Flips, Probability, and Bayes' Rule – A Deeper Dive

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

### 6. **Applying Bayes' Rule**

Now, we can calculate the posterior probability using Bayes’ Rule:

$$ P(A|B) = \\frac{P(B|A)P(A)}{P(B)} = \\frac{1 \\cdot \\frac{1}{1000}}{\\frac{1.975}{1000}} = \\frac{1}{1.975} \\approx 0.506 $$

So, given 10 heads in a row, the probability that you picked the unfair coin is about **50.6%**.

