# 使用方法

1. 安装依赖:

   ```
   python version: 3
   scipy
   numpy
   networkx
   ```

2. 修改图数据的载入和输出路径

3. `python main.py`

4. 在test文件夹下，`index.html`可以用来可视化结果（需要启动服务`python -m http.server`）

# 原理

为了建模一个图布局经过人为调整的过程，我们用一个变换矩阵来描述图中每条边是怎么变化的；在二维平面上，用旋转和平移就足以描述它。这里，我们为每条边新建一个虚拟节点，用来指示这条边旋转的方向。假设一条边原有两个节点：$v_0=(x_0, y_0), v_1=(x_1, y_1)$，新节点：
$$
v_2=v_0+\frac{(y_0-y_1,x_1-x_0)}{\sqrt{(y_0-y_1)^2+(x_1-x_0)^2}}
$$
满足$(v_2-v_0)$是$(v_1-v_0)$逆时针旋转九十度的单位向量。

![image-20200218203007336](assets/image-20200218203007336.png)

我们知道，二维平面上的旋转平移，可以通过一个$3\times3$的仿射变换矩阵来描述。我们将平移剥离，那么一个$2\times2$的矩阵，就足以描述旋转变化，我们用$Q$表示。

![](./assets/affine.png)

$$
Qv_0+d=\tilde{v_0}\\
Qv_1+d=\tilde{v_1}\\
Qv_2+d=\tilde{v_2}\\
$$

三式联立，消去$d$，我们可以得到$QV=\tilde{V}$，其中：
$$
V=[v_1-v_0, v_2-v_0]\\
\tilde{V}=[\tilde{v_1}-\tilde{v_0}, \tilde{v_2}-\tilde{v_0}]
$$
那么，
$$
\begin{align}
Q=\tilde{V}V^{-1}
=&\begin{bmatrix}
\tilde{x}_1-\tilde{x}_0 & \tilde{x}_2-\tilde{x}_0\\
\tilde{y}_1-\tilde{y}_0 & \tilde{y}_2-\tilde{y}_0
\end{bmatrix}\begin{bmatrix}
u_{00} & u_{01} \\
u_{10} & u_{11}
\end{bmatrix}\\
=&\begin{bmatrix}
-(u_{00}+u_{10})\tilde{x}_0+u_{00}\tilde{x}_1+u_{10}\tilde{x}_2 & -(u_{01}+u_{11})\tilde{x}_0+u_{01}\tilde{x}_1+u_{11}\tilde{x}_2\\
-(u_{00}+u_{10})\tilde{y}_0+u_{00}\tilde{y}_1+u_{10}\tilde{y}_2 & -(u_{01}+u_{11})\tilde{y}_0+u_{01}\tilde{y}_1+u_{11}\tilde{y}_2
\end{bmatrix}\\
=&\begin{bmatrix}
\tilde{x}_0 & \tilde{x}_1 & \tilde{x}_2 \\
\tilde{y}_0 & \tilde{y}_1 & \tilde{y}_2
\end{bmatrix}\begin{bmatrix}
-(u_{00}+u_{10}) & -(u_{01}+u_{11}) \\
u_{00} & u_{01} \\
u_{10} & u_{11}
\end{bmatrix}\\
=&\begin{bmatrix}
q_{00} & q_{01}\\
q_{10} & q_{11}
\end{bmatrix}
\end{align}
$$
我们将上述公式换一个写法：
$$
\begin{align}
\begin{bmatrix}
q_{00}\\
q_{01}\\
q_{10}\\
q_{11}
\end{bmatrix} &= \begin{bmatrix}
-(u_{00}+u_{10})\tilde{x}_0+u_{00}\tilde{x}_1+u_{10}\tilde{x}_2\\
-(u_{01}+u_{11})\tilde{x}_0+u_{01}\tilde{x}_1+u_{11}\tilde{x}_2\\
-(u_{00}+u_{10})\tilde{y}_0+u_{00}\tilde{y}_1+u_{10}\tilde{y}_2\\
-(u_{01}+u_{11})\tilde{y}_0+u_{01}\tilde{y}_1+u_{11}\tilde{y}_2
\end{bmatrix}\\
&=\begin{bmatrix}
-(u_{00}+u_{10}) & u_{00} & u_{10} & 0 & 0 & 0 \\
-(u_{01}+u_{11}) & u_{01} & u_{11} & 0 & 0 & 0 \\
0 & 0 & 0 & -(u_{00}+u_{10}) & u_{00} & u_{10} \\
0 & 0 & 0 & -(u_{01}+u_{11}) & u_{01} & u_{11} \\
\end{bmatrix}\begin{bmatrix}
\tilde{x}_0\\
\tilde{x}_1\\
\tilde{x}_2\\
\tilde{y}_0\\
\tilde{y}_1\\
\tilde{y}_2\\
\end{bmatrix}\\
&=\begin{bmatrix}
-(u_{00}+u_{10}) & 0 & u_{00} & 0 & u_{10} & 0 \\
-(u_{01}+u_{11}) & 0 & u_{01} & 0 & u_{11} & 0 \\
0 & -(u_{00}+u_{10}) & 0 & u_{00} & 0 & u_{10} \\
0 & -(u_{01}+u_{11}) & 0 & u_{01} & 0 & u_{11} \\
\end{bmatrix}\begin{bmatrix}
\tilde{x}_0\\
\tilde{y}_0\\
\tilde{x}_1\\
\tilde{y}_1\\
\tilde{x}_2\\
\tilde{y}_2\\
\end{bmatrix}
\end{align}
$$

因为我们要将源结构的变化，应用到新的结构上，我们将源结构的仿射变换矩阵写作$S$，目标结构的仿射变换矩阵写作$T$。我们需要将目标结构的边和源结构的边之间建立映射关系，我们将这种映射关系写作：
$$
M = {(s_0,t_0),(s_1,t_1),\ldots,(s_{|M|},t_{|M|})}
$$
一共$|M|$种映射方式，其中一个映射对$(s_i, t_i)$表示源结构的边$s_i$和目标结构的边$t_i$之间的映射。目标结构的每一条边都可以映射到0,1或者N条源结构的边。

我们的目标是优化：
$$
\sum_{j=0}^{|M|}||S_{s_j}-T_{t_j}||^2_F
$$
其中$S$是我们已知的，而其中$T$又能经过我们上面的变换，写成$A\tilde{x}$的形式。所以求解上述问题就是一个线性求解的问题。

