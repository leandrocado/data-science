# Lista de algoritmos
- kNN (dados estruturados - tabela - sempre em espaço linear)
- Árvode de decisão (dados estruturados - tabela - não necessariamente linear)
- Support Vector Machine (dados estruturados, e eram a melhor opção para dados não-estruturados antes de redes convolucionais - sempre em espaço linear)
- Redes neurais (super powers, espaço linear e não linear(funções de ativação))
  - Podem ter convoluções.
- Regressão logística (dados estruturados - sempre em espaço linear)

# 1 - KNN

1. Os pixels não possuem informação semântica para separar o espaço dimensional da imagem de forma a oferecer classificações precisas.

2. A complexidade computacional, no melhor dos casos é O(n log n), mas comumente é O(n), já que as distâncias devem ser computadas para cada uma das imagens na base de comparação.

3. Uma rede neural convolucional. Ela é boa para classificar dados não-estruturados por conseguir classificar classes de forma não-linear.

4. O _k_ é importante pois ele determina o número de _neighbours_ que devem ser considerados ao calcular a distância e efetuar a comparação de classificação. Caso o valor de _k_ seja muito baixo(1), há uma grande chance de ocorrer _overfitting_. Em contrapartida, um valor de _k_ muito alto(10) provavelmente implicará em _underfitting_. *Escolha da distância*.

5. 1173

```python
>>> # manhattan
>>> import numpy as np
>>> m1 = [150, 200, 125, 20, 90, 80, 220, 10, 10, 75, 50, 50]
>>> m2 = [10, 200, 115, 0, 20, 0, 100, 100, 220, 255, 2,255]
>>> a1 = np.array(m1)
>>> a2 = np.array(m2)
>>> np.sum(np.abs(a1 - a2))
>>> 1173
```

```python
>>> # euclidean
>>> np.sqrt(np.sum(np.square(a1 - a2)))
```

# 2. Classificador linear

> Por que dividir a saída do hinge-loss pelo número de imagens sendo classificadas?

> O + 1 no hinge-loss é para iniciar a correção de pesos de forma mais agressiva, dado que os valores iniciais dos pesos são próximos a zero, e portanto a margem inicial de perda é 1 / N?

1. 
```python
>>> import numpy as np
>>> x = np.array([20, 10, 25, 50])
>>> W = np.array([100, 10, 50, 50, 20, 20, 15, -50, 5, 25, 20, -10]).reshape(3, 4)
b = [12, 10, 5]
>>> c1 = np.sum(x + W[0]) + b[0]
>>> c2 = np.sum(x + W[1]) + b[1]
>>> c3 = np.sum(x + W[2]) + b[2]
>>> max(c1, c2, c3) == c1
```

2.
```python
>>> import numpy as np
>>> s = c1 + c2 + c3
>>> c1_probability = c1 / s
>>> c2_probability = c2 / s
>>> c3_probability = c3 / s
```

4.
```python
>>> import numpy as np
>>> max(0, c1 - c2) + max(0, c3 - c2)
```
