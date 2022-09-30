# RoboticsProject
Robotics Project Repository - Wildfire Detection system

##Ya se proporciona un config file llamado �prueba.yml� en la carpeta de src con la configuraci�n usada para utilizar el entrenamiento por GPU; sin embargo:

- Es necesario cambiar los paths de las carpetas de validaci�n y entrenamiento tanto para los XML como para las im�genes.
- Es necesario cambiar la ruta en la cual se guardar� el modelo entrenado


### Se necesitan las im�genes y los archivos XML con sus anotaciones, en la carpeta src se encuentra un archivo de jupyter y uno de python con la misma funcionalidad y solo es necesario usar uno de los dos:


- Los archivos deben de ser modificados en la l�nea pertinente para colocar el path de las im�genes en Seafile donde se encuentran las im�genes binarias a entrenar. Se indica qu� tan profundo se debe de seguir.
- Teniendo en cuenta que los programas se encuentran en la carpeta de src del github, correr el programa genera los archivos XML y copia las im�genes a la carpeta res, pero ya lo hace separ�ndolos en entrenamiento y validaci�n.
- Correr un archivo de python, ya sea todo el jupyter o el .py

### Si se desea entrenar la red con una gpu (RECOMENDABLE) es necesario hacer lo siguiente.

- Verificar la compatibilidad de tu gpu con cuda, seguir los pasos del siguiente foro: https://stackoverflow.com/questions/60987997/why-torch-cuda-is-available-returns-false-even-after-installing-pytorch-with
- Instalar una versi�n de cuda compatible con tu gpu y sistema operativo. https://pytorch.org/get-started/locally/
- Instalar nanodet: https://github.com/RangiLyu/nanodet#install, sin embargo es necesario modificar un comando para que todo est� correcto.
- Comando a modificar: conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c conda-forge, en el par�metro de cudatoolkit=11.1 debes cambiar el valor de 11.1 a la versi�n que tengas instalada en tu computadora. 
- Verificar que todo est� correctamente instalado: 
  - Inicializar la m�quina virtual de nanodet.
  - Colocarse en el path de nanodet
  - Ejecutar python y correr los siguientes comandos:

  ```shell script
  import troch
  torch.cuda.is_available()
  ```
	El resultado deber�a dar True.
Seguir los pasos de entrenamiento de nanodet : 
Para poder entrenar la red, es necesario crear un archivo .xml, se proporciona un archivo de ejemplo. Dentro de este archivo se hacen las configuraciones deseadas, para modificar correctamente este archivo referirse al siguiente link, paso 2: https://github.com/RangiLyu/nanodet#how-to-train
En la parte de device colocar el identificador de gpu que deseas, si tienes una sola gpu colocar: [0].
Finalmente correr el siguiente comando: 
```shell script
python tools/train.py CONFIG_FILE_PATH
 ```
En caso de sufrir un error de memoria al iniciar el entrenamiento, se debe modificar el tama�o de batch en la l�nea 91 del config file.

### Para entrenar la red usando CPU es necesario realizar pasos similares pero no tantos:

- Instalar nanodet: https://github.com/RangiLyu/nanodet#install, sin embargo es necesario modificar un comando para que todo est� correcto.
- Comando a modificar: conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c conda-forge, en su lugar se utiliza: conda install pytorch torchvision torchaudio cpuonly -c pytorch
- Verificar que todo est� correctamente instalado: 
  - Inicializar la m�quina virtual de nanodet.
  - Colocarse en el path de nanodet
- Seguir los pasos de entrenamiento de nanodet : 
- Para poder entrenar la red, es necesario crear un archivo .xml, se proporciona un archivo de ejemplo. Dentro de este archivo se hacen las configuraciones deseadas, para modificar correctamente este archivo referirse al siguiente link, paso 2: https://github.com/RangiLyu/nanodet#how-to-train
- En la parte de device colocar el identificador de cpu: -1.
- Finalmente correr el siguiente comando: 
  ```shell script
  python tools/train.py CONFIG_FILE_PATH
  ```
