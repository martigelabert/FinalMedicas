# Práctica Final Imagenes medicas
Los dos ejercicios están separados en dos .py diferentes.

## Ejercicio 1
En el caso del primer ejercicio los gifs se guardan dentro de la carpeta de result. Y dependiendo del tipo de proyección inicial que se elija en las opciones del script, aparecerá en su carpeta correspondiente.
En este ejercicio se requería:
- Obtener los datos de DICOM.
- Ordenar las CTs del paciente.
- Obtener la segmentación de las 4 regiones.
- Ordenas la segmentación.
- Generar una animación de una proyección saggital-coronal rotando.
- Aplicar las zonas segmentadas y visualizarlas.

## Ejercicio 2
Para visualizar los resultados del 2 hay que ejecutar el proyecto. En este ejercicio se pedia realizar una tarea de coregistración. (Si se requiere mostrar las landmarks en un espacio 3D setear a true la variable _view3D)
En este ejercicio se requería:
- Realizar un proceso parecido al ejercicio anterior.
- Cargar las imagenes del paciente, phantom y atlas.
- Ajustar las dimensiones del paciente.
- Generar landmarks.
- Encontrar los parametros optimos para realizar la corregistración empleando los landmarks. 
- Mostrar resultados corregistrando el paciente y las imagenes del phantom
- Mostrar una segmentación del thalamus en el paciente.

Por lo general dependiendo del número de landmarks que seleccionemos
```
    landmarks_ref = images_phantom[::9, ::9, ::9].reshape(-1,3) # cambiarlo por números divisibles por 3, para poder transformar a coord tipo voxel
    landmarks_input = processedCT[::9, ::9, ::9].reshape(-1,3)

```
Tenemos que tener en cuenta que repercutira en nuestro coste computacional, pero podremos obtener mejores resultados. 
