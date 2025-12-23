
# Preguntas sobre el ejemplo de clasificación de imágenes con PyTorch y MLP

## 1. Dataset y Preprocesamiento
- ¿Por qué es necesario redimensionar las imágenes a un tamaño fijo para una MLP?
Porque MLP trabaja con vectores de tamaño fijo, asi coincide la matriz de pesos y si no habria error de dimension.

- ¿Qué ventajas ofrece Albumentations frente a otras librerías de transformación como `torchvision.transforms`?
Es más rapida, mejora el rendimiento, tiene muchas tranformaciones para imagenes y se adapta facil.

- ¿Qué hace `A.Normalize()`? ¿Por qué es importante antes de entrenar una red?
Normaliza los valores de los pixeles de una imagen antes de entrar en la red neuronal, quedan centrados en cero, esto genera una distribucion mejor para el entrenamiento, reduce problemas con los gradientes y mejora la estabilidad.

- ¿Por qué convertimos las imágenes a `ToTensorV2()` al final de la pipeline?
Transforma la imagen para que sea compatible con Pytorch


## 2. Arquitectura del Modelo
- ¿Por qué usamos una red MLP en lugar de una CNN aquí? ¿Qué limitaciones tiene?
Se usa como baseline, para analizar las caracteristicas de las imagenes sin un modelo tan complejo. Pierde info espacial, tiene mas parametros, no generaliza tan bien. 

- ¿Qué hace la capa `Flatten()` al principio de la red?
Convierte la imagen en un vector para que se pueda procesar.

- ¿Qué función de activación se usó? ¿Por qué no usamos `Sigmoid` o `Tanh`?
Se usa ReLu, es una funcion no lineal. No satura

- ¿Qué parámetro del modelo deberíamos cambiar si aumentamos el tamaño de entrada de la imagen?
Input_size


## 3. Entrenamiento y Optimización
- ¿Qué hace `optimizer.zero_grad()`?
Limpia los gradientes anteriores y evita que gradientes de distintos batches se mezclen.

- ¿Por qué usamos `CrossEntropyLoss()` en este caso?
Porque es un problema de clasificacion multiclase.

- ¿Cómo afecta la elección del tamaño de batch (`batch_size`) al entrenamiento?
Si el batch es mas chico el gradiente es ma ruidoso, el entrenamiento es mas lento y las metricas son inestables. Si el batch es grande el gradiente es mas estable pero no generaliza tan bien y rquiere mas memoria.

- ¿Qué pasaría si no usamos `model.eval()` durante la validación?
Las capas se comportan incorrectamente y genera metricas de evaluacion poco confiables.

## 4. Validación y Evaluación
- ¿Qué significa una accuracy del 70% en validación pero 90% en entrenamiento?
Overfitting 

- ¿Qué otras métricas podrían ser más relevantes que accuracy en un problema real?
Score, ROC

- ¿Qué información útil nos da una matriz de confusión que no nos da la accuracy?
Que clases se confunden, cuantos falsos positivos y negativos hay, por ejemplo el accuracy es 85% pero en la matriz se ve que una clases nunca se predice bien. 

- En el reporte de clasificación, ¿qué representan `precision`, `recall` y `f1-score`?
Precision: cuantas fueron correctas
Recall: cuantos falsos negativos 
F1-score: balance entre precision y recall

## 5. TensorBoard y Logging
- ¿Qué ventajas tiene usar TensorBoard durante el entrenamiento?
Permite monitorear y enender que esta pasado dentro del entrenamiento en tiempo real, se puede ver como evolucionan las metricas.

- ¿Qué diferencias hay entre loguear `add_scalar`, `add_image` y `add_text`?
`add_scalar`: registra valores numericos, sirve para curvas de entrenamiento.
`add_image`: interpretacion visual
`add_text`: reusltados

- ¿Por qué es útil guardar visualmente las imágenes de validación en TensorBoard?
Permite verificar el correcto procesamiento de los datos e interpretar visualmente los problemas del modelo

- ¿Cómo se puede comparar el desempeño de distintos experimentos en TensorBoard?


## 6. Generalización y Transferencia
- ¿Qué cambios habría que hacer si quisiéramos aplicar este mismo modelo a un dataset con 100 clases?
DEbemos cambiar el numero de neurosas de salida, agregar mas capas 

- ¿Por qué una CNN suele ser más adecuada que una MLP para clasificación de imágenes?
aprende de patrones locas, es invariante a traslaciones y tiene menos parametros

- ¿Qué problema podríamos tener si entrenamos este modelo con muy pocas imágenes por clase?
ss puede producir un overfitting 

- ¿Cómo podríamos adaptar este pipeline para imágenes en escala de grises?
Image.open(path).convert("L")


## 7. Regularización

### Preguntas teóricas:
- ¿Qué es la regularización en el contexto del entrenamiento de redes neuronales?
sirve para disminuir el sobreajuste y que el modelo aprenda patrones generales y no memorice.

- ¿Cuál es la diferencia entre `Dropout` y regularización `L2` (weight decay)?
Dropout: agarra neuronas al azar y se aplica solo en entrenamiento.
L2: penaliza pesos grandes y actua siempre.

- ¿Qué es `BatchNorm` y cómo ayuda a estabilizar el entrenamiento?
Notmaliza las activaciones intermedias, hace el entrenamiento mas estable.

- ¿Cómo se relaciona `BatchNorm` con la velocidad de convergencia?
Se pueden usar learning rate mas grandes, el gradiente es mas estable y el modelo converge en menos epocas.

- ¿Puede `BatchNorm` actuar como regularizador? ¿Por qué?

- ¿Qué efectos visuales podrías observar en TensorBoard si hay overfitting?
curvas divergentes de train loss y val loss, accuracy de train sube y la de val baja, metricas oscilan mucho.

- ¿Cómo ayuda la regularización a mejorar la generalización del modelo?
Evita que se ejute al ruido

### Actividades de modificación:
1. Agregar Dropout en la arquitectura MLP:
   - Insertar capas `nn.Dropout(p=0.5)` entre las capas lineales y activaciones.
   - Comparar los resultados con y sin `Dropout`.
   SIN DROPOUT:
   Epoch 10:
  Train Loss: 0.9547, Accuracy: 62.70%
  Val   Loss: 1.1691, Accuracy: 55.80%
Epoch 20:
  Train Loss: 0.7129, Accuracy: 71.02%
  Val   Loss: 1.3480, Accuracy: 55.80%

  CON DROPOUT
  Epoch 10:
  Train Loss: 1.0185, Accuracy: 63.56%
  Val   Loss: 1.1623, Accuracy: 58.56%
Epoch 20:
  Train Loss: 0.7470, Accuracy: 71.31%
  Val   Loss: 1.2319, Accuracy: 59.12%

Coo el dropout agregra neuronas al azar hay menos riesgo de memorizar, se puedever que sin dropout aumenta el accuracy pero no mejora validacion , en cambio con dropout validacion va subiendo de a poco.

2. Agregar Batch Normalization:
   - Insertar `nn.BatchNorm1d(...)` después de cada capa `Linear` y antes de la activación:
     ```python
     self.net = nn.Sequential(
         nn.Flatten(),
         nn.Linear(in_features, 512),
         nn.BatchNorm1d(512),
         nn.ReLU(),
         nn.Dropout(0.5),
         nn.Linear(512, 256),
         nn.BatchNorm1d(256),
         nn.ReLU(),
         nn.Dropout(0.5),
         nn.Linear(256, num_classes)
     )
     ```

     Con la combinacion de batch normalization + dropout=0.5 se ve que el entrenamiento es ruidoso porque esta sobre regularizado, se observa cuando el accuracy sube y baja.

3. Aplicar Weight Decay (L2):
   - Modificar el optimizador:
     ```python
     optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
     ```
     con esta combinacion el entrenamiento es aun mas estable y mejora la generalizacion, el modelo aprende un poco mas lento pero mas consistentemente.

4. Reducir overfitting con data augmentation:
   - Agregar transformaciones en Albumentations como `HorizontalFlip`, `BrightnessContrast`, `ShiftScaleRotate`.
Aca se puede ver que con estas transformaciones el train es mas bajo pero validacion se acerca mas a ese valor lo que significa que esta entrenando mejor.


5. Early Stopping (opcional):
   - Implementar un criterio para detener el entrenamiento si la validación no mejora después de N épocas.

### Preguntas prácticas:
- ¿Qué efecto tuvo `BatchNorm` en la estabilidad y velocidad del entrenamiento?
Mejoro la estabilidad en cuanto a la optimizacion pero velocidad de entrenamieto no aumento, y al combinar con dropout y L2 se hizo mas lento.

- ¿Cambió la performance de validación al combinar `BatchNorm` con `Dropout`?
Es mas inestable cuando se combina.

- ¿Qué combinación de regularizadores dio mejores resultados en tus pruebas?
BatchNorm + Dropout(0.5) + L2 (weight decay 1e-4)
el mejor accuracy es de 61,88% y no hay picos fuertes 

- ¿Notaste cambios en la loss de entrenamiento al usar `BatchNorm`?
Si, baja mas contrladamente, la train loss inicial es mas alta es decir que tarda mas en acomodarse, y que hay menos correlacion entre train loss y val accuracy

## 8. Inicialización de Parámetros

### Preguntas teóricas:
- ¿Por qué es importante la inicialización de los pesos en una red neuronal?
Porque determina como fluye la informacion y el gradiente al inciio del entrenamiento

- ¿Qué podría ocurrir si todos los pesos se inicializan con el mismo valor?
Problema de simetria 

- ¿Cuál es la diferencia entre las inicializaciones de Xavier (Glorot) y He?
Xavier: es para activaciones Sigmoid y Tanh, mantiene la varianza de activciones constante entre capas y asume activaciones simetricas alrededor de 0.
He: es para ReLu, tiene en cuenta que ReLu apaga la mitad de las activaciones y permite varianza mayor para compensar eso.

- ¿Por qué en una red con ReLU suele usarse la inicialización de He?
Porque ReLu apaga valores negativos y eso reduce la varianza de activaciones, con He aumenta la varianza de los pesos, compensa la perdida de activaciones negativas, mantiene el flujo de gradiente estable.

- ¿Qué capas de una red requieren inicialización explícita y cuáles no?
la capa lineal y la convolucional requieren inicializacion explicita  porque son las que aprenden pesos. 
La ReLu, dropout, flatten, MaxPool y batchnorm no necesitan


### Actividades de modificación:
1. Agregar inicialización manual en el modelo:
   - En la clase `MLP`, agregar un método `init_weights` que inicialice cada capa:
     ```python
     def init_weights(self):
         for m in self.modules():
             if isinstance(m, nn.Linear):
                 nn.init.kaiming_normal_(m.weight)
                 nn.init.zeros_(m.bias)
     ```

2. Probar distintas estrategias de inicialización:
   - Xavier (`nn.init.xavier_uniform_`)
   - He (`nn.init.kaiming_normal_`)
   - Aleatoria uniforme (`nn.init.uniform_`)
   - Comparar la estabilidad y velocidad del entrenamiento.

   Xavier genera un resultado mas estable con mejor generalizacion y la velocidad es estable y suave. He es mas lento al inicio pero mas consistente que uniform, y uniform arranca bien pero converge lento.

3. Visualizar pesos en TensorBoard:
   - Agregar esta línea en la primera época para observar los histogramas:
     ```python
     for name, param in model.named_parameters():
         writer.add_histogram(name, param, epoch)
     ```

     TensorBoard 2.20.0 at http://localhost:6021/

### Preguntas prácticas:
- ¿Qué diferencias notaste en la convergencia del modelo según la inicialización?
uniform es la que convergia mas lento

- ¿Alguna inicialización provocó inestabilidad (pérdida muy alta o NaNs)?
la He combinada con batchnorm genero mas inestabilidad.

- ¿Qué impacto tiene la inicialización sobre las métricas de validación?
si es una buena inicializacion generaliza mejor, hay menos gap entre train y validacion y disminuye el overfitting.

- ¿Por qué `bias` se suele inicializar en cero?
asi se centra la activacion y permite que el modelo aprenda de una base nuetra.