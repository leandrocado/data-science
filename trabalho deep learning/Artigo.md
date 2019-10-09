---
title: "Trabalho final - Deep Learning I"
author: Alan Aguirre, Leandro Cadó, Martin Jungblut, Matheus Prola
geometry: margin=1.5cm
output: pdf_document
---

# Introdução

Para nosso trabalho, utilizamos o dataset ["Hand Gesture Recognition Database"](https://www.kaggle.com/gti-upm/leapgestrecog), que contém 10 gestos manuais demonstrados por 10 pessoas diferentes, sendo 5 homens e 5 mulheres. Para cada indíviduo, 2000 imagens foram coletadas, 200 imagens para cada gesto. A resolução das imagens é de 640 x 240 pixels.

O dataset está disposto dentro de um diretório principal, dividido em subdiretórios, um para cada pessoa, com diretórios internos para cada gesto.

_**Figura 1.**_

Escolhemos este dataset por ser composto de imagens, o que nos permite explorar melhor a natureza de redes neurais convolucionais. Também foi escolhido por não ser um dataset muito grande, o que torna o ciclo de treino e verificação moderadamente curto em hardware _mid-tier_.

Dessa forma, conseguimos explorar diferentes topologias e alcançarmos uma boa acurácia, durante vários momentos das diferentes etapas de validação, assim como em uma fase final de teste para cada topologia e combinação de hiperparâmetros que foi abordada.

# Desenvolvimento

Nossas primeiras tentativas partiram de uma rede Resnet50 pré-treinada, utilizando 10 épocas e um tamanho de _batch_ 4. Cada imagem foi redimensionada para um tamanho de 32 x 32 pixels.
Para a normalização, utilizamos a média e um desvio padrão de 0.5.

Durante este momento, a rede conseguia classificar corretamente 90% das imagens durante a validação, porém com um tempo de treinamento muito alto. Percebemos que ao utilizarmos um _BATCH_SIZE_ muito pequeno, o processo de treino era consideravelmente lento. Concluímos que, ao utilizar um _BATCH_SIZE_ muito pequeno, tanto a CPU como a GPU ficavam muito tempo ociosas, enquanto os dados trafegavam pelo barramento PCI Express em cada ciclo do treinamento.

Aumentamos o _BATCH_SIZE_ para 64, e em alguns minutos conseguimos executar 10 épocas em uma RTX 2060, assim como no Google Collab.

A cada 10 iterações dentro de uma época, mostramos o valor do custo para o subconjuntos de treinamento e validação.

Após esse teste inicial, decidimos experimentar três diferentes redes pré-treinadas como ponto de partida: Resnet50, Resnet101 e GoogleNet.

Para cada uma das redes, utilizamos a mesma normalização, que é a normalização com as quais elas foram préviamente treinadas no PyTorch.

Médias: 0.485, 0.456, 0.406. Desvios padrão: 0.229, 0.224, 0.225.

Além disso, as imagens foram redimensionadas para 256 x 256 pixels, e capturamos somente o centro das mesmas para os treinamentos, de tamanho 224 x 224 pixels.

Também utilizamos a mesma topologia sequencial, que foi adicionada na fase final de classificação de cada uma das redes.

- Linear(tamanho da saída da camada anterior, 512)
- ReLU
- Dropout(0.2)
- Linear(512, 10)
- LogSoftmax(dim=1)

Como algoritmo de otimização, utilizamos Adam, por utilizar pouca memória, ser computacionalmente eficiente e possuir hiperparâmetros de fácil compreensão. Utilizamos um _learning rate_ de 0.003.

Dividimos os dados em subconjuntos de treinamento(81%), validação(9%) e teste(10%), e aplicamos um resampling nos subconjuntos de treinamento e validação no ciclo de cada época, a fim de maximizar o aprendizado e diminuir os riscos de _overfitting_.

Decisões tomadas, vamos aos testes:

# Resnet50

Congelamos todas camadas anteriores da rede, e alteramos somente a camada _fully connected_(FC), utilizando a topologia demonstrada préviamente, com uma entrada de 2048 parâmetros na primeira camada linear.

No início do primeira época, o _loss_ para o subconjunto de treinamento foi 3.11, e 2.19 para o subconjunto de validação. A precisão de classificação foi de 33%.
Com isso, podemos verificar que inicialmente a rede não consegue classificar as imagens do dataset com uma boa acurácia.

Ao final da primera época, o _loss_ para o subconjunto de treinamento foi de 0.088, e 0.064 para o subconjunto de validação. A precisão de classificação foi de 98.3%.
Notamos que, mesmo em apenas uma época, com os hiperparâmetros que escolhemos, o tamanho da entrada, e a topologia escolhida, conseguimos uma acurácia satisfatória.

Podemos verificar a curva decrescente do custo através da épocas, para os subconjuntos de treinamento e validação, exemplificados na _**Figura 2**_.

Ao final da décima e última época, o _loss_ para o subconjunto de treinamento foi de 0.015, e 0.004 para o subconjunto de validação. A precisão de classificação foi de 99.9%.
Aqui percebemos o real poder da rede, diminuindo ainda mais os valores dos custos, e aumentando em 1.6% a precisão de classificações subsequentes.

Ao testarmos essa rede com dados desconhecidos, pertencentes ao subconjunto de teste, podemos verificar que a rede obteve uma acurácia de 99.1%.

# Resnet101

Como no teste anterior, congelamos todas camadas anteriores da rede, e alteramos somente a camada _fully connected_(FC), utilizando a mesma topologia da Resnet50.

No início do primeira época, o _loss_ para o subconjunto de treinamento foi 2.74, e 2.05 para o subconjunto de validação. A precisão de classificação foi de 39%.
Esse resultado, embora positivo, não é suficientemente diferente do resultado da Resnet50 para conseguirmos determinar algum comportamento nesse momento.

Ao final da primera época, o _loss_ para o subconjunto de treinamento foi de 0.064, e 0.040 para o subconjunto de validação. A precisão de classificação foi de 99%.
Comparando com a Resnet50, o resultado obtido nessa época foi superior.

Podemos verificar a curva decrescente do custo através da épocas, para os subconjuntos de treinamento e validação, exemplificados na _**Figura 3**_.

Ao final da décima e última época, o _loss_ para o subconjunto de treinamento foi de 0.029, e 0.006 para o subconjunto de validação. A precisão de classificação foi de 99.8%.

Ao testarmos essa rede com dados do subconjunto de teste, verificamos que a rede obteve uma acurácia de 98.7%.

# GoogleNet

Como nos testes anterior, congelamos todas camadas anteriores da rede, e alteramos somente a camada _fully connected_(FC), utilizando a mesma topologia das redes anteriores, porém com uma entrada de 1024 parâmetros na primeira camada linear.

No início do primeira época, o _loss_ para o subconjunto de treinamento foi 2.16, e 1.78 para o subconjunto de validação. A precisão de classificação foi de 34.9%.
E ao final da primera época, o _loss_ para o subconjunto de treinamento foi de 0.074, e 0.032 para o subconjunto de validação. A precisão de classificação foi de 99.4%.

Podemos verificar a curva decrescente do custo através da épocas, para os subconjuntos de treinamento e validação, exemplificados na _**Figura 4**_.

Ao final da décima e última época, o _loss_ para o subconjunto de treinamento foi de 0.024, e 0.004 para o subconjunto de validação. A precisão de classificação foi de 100%.

Ao testarmos essa rede com dados do subconjunto de teste, verificamos que a rede obteve uma acurácia de 95.3%.

# Conclusão

Comparando as redes Resnet50 e Resnet101, concluímos que a eficiência de ambas se mostrou muito similar. Atribuímos a pequena diferença de acurácia à camada _dropout_ de nossa topologia.

A rede GoogleNet mostrou-se eficiente durante o ciclo de treinamento/validação, porém mostrou uma performance inferior ao classificar dados desconhecidos. Pensamos que talvez isso se deva ao fato de sua primeira camada linear possuir menos parâmetros do que as Resnet.

Notamos também que todas redes pré-treinadas iniciaram um aumento significativo no valor de custo nas primeiras iterações, porém que rapidamente decresce de forma exponencial.

![Figura1](https://user-images.githubusercontent.com/50786369/66450599-83f8e480-ea2f-11e9-820f-2a6ac7771852.jpg)

![Figura2](https://user-images.githubusercontent.com/50786369/66450086-9f62f000-ea2d-11e9-9064-f1e591256852.JPG)

![Figura3](https://user-images.githubusercontent.com/50786369/66450087-9ffb8680-ea2d-11e9-9df5-2e99c8b82ef6.JPG)

![Figura4](https://user-images.githubusercontent.com/50786369/66450088-9ffb8680-ea2d-11e9-8805-1aa60708140a.JPG)
