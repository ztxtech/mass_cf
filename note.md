# Mass Distribution的特征函数


## 矩母函数（Moment Generating Function, MGF）
### 定义
对于随机变量 $X$，其矩母函数定义为：  
$$ 
M(t) = E\left[e^{tX}\right] = \sum_{x} e^{tx} \cdot p(x) \quad (\text{离散情形}),
$$  
其中：  
- $t$ 是实数；  
- $p(x)$ 是 $X$ 的概率质量函数；  
- 求和遍历 $X$ 的所有可能取值 $x$。

### 伯努利分布的矩母函数
设 $X \sim \text{Bernoulli}(p)$，其矩母函数为：  
$$ 
M(t) = e^{t \cdot 0} \cdot (1-p) + e^{t \cdot 1} \cdot p = (1-p) + p \cdot e^{t}. 
$$
对比特征函数 $\varphi(\omega) = (1-p) + p e^{it}$，可见二者形式相似，但参数 $t$ 的性质不同。


### 主要性质
#### 存在性
- 存在条件：$M(t)$ 仅在 $t$ 的某个邻域内存在（如 $|t| < a$），因为 $e^{tx}$ 可能随 $x$ 增大而发散。  
- 反例：柯西分布的 MGF 不存在（但特征函数存在）。

#### 矩的生成
通过对 $M(t)$ 求导并在 $t=0$ 处取值，可直接得到各阶矩：  
$$ 
E[X^n] = \left. \frac{d^n M(t)}{dt^n} \right|_{t=0}. 
$$  
示例：  
- 一阶导数：$M'(t) = p e^t$，故 $E[X] = M'(0) = p$；  
- 二阶导数：$M''(t) = p e^t$，故 $E[X^2] = M''(0) = p$。

#### 独立和的简化
若 $X_1, X_2, \dots, X_n$ 独立，则它们的和的 MGF 等于各自 MGF 的乘积：  

$$ 
M_{X_1+X_2+\cdots+X_n}(t) = M_{X_1}(t) \cdot M_{X_2}(t) \cdot \dots \cdot M_{X_n}(t).
$$


#### 唯一性
在 $M(t)$ 的收敛区间内，矩母函数唯一确定概率分布。但需注意，某些分布可能具有相同的矩序列但不同的分布（如对数正态分布的变体），此时 MGF 可能不存在或不唯一。

## 特征函数（Characteristic Function, $\varphi$）

### 定义  
对于离散型随机变量$ X $，其特征函数$ \varphi(\omega) $ 定义为：  
$$
\varphi(\omega) = E\left[e^{i \omega X}\right] = \sum_{x} e^{i \omega x} \cdot p(x)
$$  
其中：  
-$ t $ 是实数，$ i $ 是虚数单位（$ i^2 = -1 $）；  
-$ p(x) $ 是$ X $ 的概率质量函数（即$ P(X = x) = p(x) $）；  
- 求和遍历$ X $ 的所有可能取值$ x $。

### 示例：伯努利分布的特征函数  
设$ X $ 服从参数为$ p $ 的伯努利分布（即$ X $ 取值为 0 或 1，概率分别为$ 1-p $ 和$ p $），则其特征函数为：  
$$
\varphi(\omega) = e^{it \cdot 0} \cdot (1-p) + e^{it \cdot 1} \cdot p = (1-p) + p \cdot e^{it}.
$$

### 主要性质  
1. 唯一性：离散分布与其特征函数一一对应，即不同的离散分布具有不同的特征函数。  
2. 矩的计算：通过对特征函数求导可得到随机变量的各阶矩。例如，一阶矩（期望）为$ E[X] = \varphi'(0) $。  
3. 独立和的简化：若$ X_1, X_2, \dots, X_n $ 独立，则它们的和的特征函数等于各自特征函数的乘积。

## 特征函数 vs 矩母函数

### 定义不同
- 特征函数：  
  对随机变量 $ X $，其特征函数定义为：  
  $$
  \varphi(\omega) = E\left[e^{i \omega X}\right] = \sum_{x} e^{i \omega x} \cdot p(x) \quad (\text{离散情形})
  $$  
  其中 $ t $ 是实数，$ i $ 是虚数单位。

- 矩母函数：  
  矩母函数定义为：  
  $$
  M(t) = E\left[e^{tX}\right] = \sum_{x} e^{tx} \cdot p(x) \quad (\text{离散情形})
  $$  
  其中 $ t $ 是实数。

核心区别：特征函数中引入了虚数 $ i $，而矩母函数是纯实数运算。


### 存在性不同
- 特征函数：  
  对所有随机变量都存在，因为 $ |e^{i \omega x}| = 1 $，保证了级数或积分绝对收敛。

- 矩母函数：  
  仅在 $ t $ 的某个邻域内存在（例如，当 $ X $ 的矩生成函数在包含原点的区间内收敛时）。  
  举例：柯西分布的矩母函数不存在，但特征函数存在。

### 矩的计算方式不同
- 特征函数：  
  通过对 $ \varphi(\omega) $ 求导并在 $ \omega=0 $ 处取值，可得到各阶矩。例如：  
  $$
  E[X^n] = \left. \frac{d^n \varphi(\omega)}{d\omega^n} \right|_{\omega=0} \cdot \frac{1}{i^n}.
  $$

- 矩母函数：  
  直接对 $ M(t) $ 求导并在 $ t=0 $ 处取值，即可得到矩：  
  $$
  E[X^n] = \left. \frac{d^n M(t)}{dt^n} \right|_{t=0}.
  $$

优势：特征函数在处理高阶矩或复杂分布时更稳定，因为其始终存在；而矩母函数可能因发散而无法使用。


### 应用场景不同
- 特征函数：  
  - 常用于理论分析，如证明中心极限定理、研究独立随机变量和的分布。  
  - 对周期性分布（如泊松分布）的分析更方便。

- 矩母函数：  
  - 更直观地与矩相关，适合直接计算期望、方差等。  
  - 在处理某些统计模型（如指数族分布）时更简洁。


### 总结对比
| 特性         | 特征函数                          | 矩母函数                          |
|------------------|---------------------------------------|---------------------------------------|
| 定义         | $ \varphi(\omega) = E[e^{i \omega X}] $         | $ M(t) = E[e^{tX}] $                |
| 存在性       | 始终存在                              | 仅在特定区间内存在                    |
| 矩的计算     | 通过虚数导数，公式稍复杂              | 直接求导，公式更简单                  |
| 应用         | 理论分析、极限定理                    | 矩的直接计算、特定统计模型            |

## 特征函数和傅里叶变换的关系

### 定义上的直接对应
- 傅里叶变换的一般形式为：  
  $$
  \mathcal{F}(\omega) = \int_{-\infty}^{\infty} f(x) \cdot e^{-i\omega x} \, dx,
  $$  
  其中 $ f(x) $ 是定义在实数域上的函数，$ \omega $ 是频率参数。

- 特征函数的定义为：  
  $$
  \varphi(\omega) = E\left[e^{i \omega X}\right] = \int_{-\infty}^{\infty} e^{i \omega x} \cdot p(x) \, dx \quad (\text{连续情形}),
  $$  
  其中 $ p(x) $ 是随机变量 $ X $ 的概率密度函数，$ t $ 是实数参数。

核心联系：  
若将概率密度函数 $ p(x) $ 视为傅里叶变换的输入函数 $ f(x) $，并将频率参数 $ \omega $ 替换为 $ -\omega $，则特征函数可表示为：  
$$
\varphi(\omega) = \mathcal{F}(-\omega).
$$  
因此，特征函数是概率分布 $ p(x) $ 的傅里叶变换在频率取负值时的结果。

### 离散情形的对应
对于离散型随机变量 $ X $，其特征函数为：  
$$
\varphi(\omega) = \sum_{x} e^{i \omega x} \cdot p(x),
$$  
这对应于离散傅里叶变换（Discrete Fourier Transform, DFT），即将概率质量函数 $ p(x) $ 作为离散信号进行傅里叶变换。

### 性质的一致性
傅里叶变换的许多性质在特征函数中同样成立，例如：
- 线性性质：若 $ X $ 和 $ Y $ 独立，则 $ \varphi_{X+Y}(t) = \varphi_X(t) \cdot \varphi_Y(t) $，这与傅里叶变换中“卷积的傅里叶变换等于傅里叶变换的乘积”一致。
- 唯一性：傅里叶变换在一定条件下是唯一的，这与特征函数唯一确定概率分布的性质一致。
- 逆变换：对于连续型随机变量，概率密度函数 $ p(x) $ 可通过对特征函数进行逆傅里叶变换得到：  
  $$
  p(x) = \frac{1}{2\pi} \int_{-\infty}^{\infty} \varphi(\omega) \cdot e^{-i \omega x} \, d\omega.
  $$

## 特征函数求高阶矩的细节
### 一阶导数的计算
对 $ \varphi(t) $ 关于 $ t $ 求导时，可交换求和与求导的顺序（需满足一致收敛条件，此处默认成立）：  
$$
\varphi'(t) = \frac{d}{dt} \sum_{x} e^{itx} \cdot p(x) = \sum_{x} \frac{d}{dt} \left( e^{itx} \cdot p(x) \right).
$$  
对每一项求导：  
$$
\frac{d}{dt} e^{itx} = i x \cdot e^{itx},
$$  
因此：  
$$
\varphi'(t) = \sum_{x} i x \cdot e^{itx} \cdot p(x).
$$

### 在 $ t = 0 $ 处求值
将 $ t = 0 $ 代入一阶导数，由于 $ e^{i \cdot 0 \cdot x} = 1 $，可得：  
$$
\varphi'(0) = \sum_{x} i x \cdot p(x) = i \sum_{x} x \cdot p(x) = i \cdot E[X].
$$  
这表明 特征函数的一阶导数在 $ t=0 $ 处的值与随机变量的期望 $ E[X] $ 相关。

### 高阶导数的推广
类似地，可计算 $ n $ 阶导数。例如，二阶导数为：  
$$
\varphi''(t) = \frac{d^2}{dt^2} \sum_{x} e^{itx} \cdot p(x) = \sum_{x} (i x)^2 \cdot e^{itx} \cdot p(x),
$$  
在 $ t = 0 $ 处：  
$$
\varphi''(0) = \sum_{x} (i x)^2 \cdot p(x) = i^2 \sum_{x} x^2 \cdot p(x) = - E[X^2].
$$  
一般地，$ n $ 阶导数为：  
$$
\varphi^{(n)}(0) = i^n \cdot E[X^n].
$$

### 示例：伯努利分布的特征函数求导
设 $ X \sim \text{Bernoulli}(p) $，其特征函数为：  
$$
\varphi(t) = (1-p) + p e^{it}.
$$  
- 一阶导数：  
  $$
  \varphi'(t) = 0 + p \cdot i e^{it} = i p e^{it},
  $$  
  在 $ t = 0 $ 处：  
  $$
  \varphi'(0) = i p \cdot 1 = i p = i \cdot E[X] \quad (\text{因为 } E[X] = p).
  $$

- 二阶导数：  
  $$
  \varphi''(t) = i p \cdot i e^{it} = -p e^{it},
  $$  
  在 $ t = 0 $ 处：  
  $$
  \varphi''(0) = -p = - E[X^2] \quad (\text{因为 } E[X^2] = p).
  $$

## 特殊的基本概率指派

### 目前已知的
1. 全集为1: $$m(\Omega)=1$$
2. 均匀分布: $$m(i) = \frac{1}{2^n-1}$$
3. 最大熵分布: $$m(i) = \frac{|\mathcal{E}(i)|}{\sum_{j \subseteq \mathcal{E}(\Omega)} |\mathcal{E}(j)|}$$

### 函数生成的新模型
1. 类型1: $$ m(i) = \frac{f(bin(i))}{\sum_{j \subseteq \mathcal{E}(\Omega)} f(bin(j))}$$
2. 类型2: 
$$ m(i) = \frac{f(bin(i))*|\mathcal{E}(i)|}{\sum_{j \subseteq \mathcal{E}(\Omega)} f(bin(j))*|\mathcal{E}(i)|}$$

$$ \frac{m(i)}{|\mathcal{E}(i)|} = \frac{f(bin(i))}{\sum_{j \subseteq \mathcal{E}(\Omega)} f(bin(j))*|\mathcal{E}(i)|}$$

$$ H_g = -\sum_{i \in \mathcal{E}(\Omega)} m(i)*log_2(\frac{m(i)}{|\mathcal{E}(i)|}) $$

## 程若兰师姐的傅里叶变换
1. 常规形式
$$F(\omega) = \sum_{a \in \mathcal{E}(\Omega)} m(a)*e^{-i*\sum_{\omega_c \in (\omega_{bin(a)} \cap \omega)} w_c}$$
2. DS规则
$$F_1(\omega)* F_2(w) = \sum_{a \in \mathcal{E}(\Omega)} \sum_{b \in \mathcal{E}(\Omega)} m_1(a)*m_1(b)*e^{-i*\sum_{\omega_c \in ((\omega_{bin(a)} \cup \omega_{bin(b)}) \cap \omega)} w_c}$$
**主要思想**
保持两个mass function的外积和的存在。