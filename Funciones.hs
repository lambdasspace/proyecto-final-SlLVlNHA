module Funciones where

import System.Random
import Data.List
import Data.List.Split
import System.IO

-- Archivos que contienen los datos de entrenamiento y los pesos respectivamente
baseDatos = "poker-hand-testing.data"
datosVal = "poker-hand-testing.data"
pesos0 = "pesosw0final.data"
pesos1 = "pesosw1final.data"

-- +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
-- + FUNCIONES AUXILIARES                                                    +
-- +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

-- Función que nos permite crear los pesos con valores entre 0 y 1 para cada 
-- entrada de manera aleatoria.
-- l: numero de neuronas en la capa oculta
-- n: número de entradas
-- m: cambia el generador para que no nos de los mismos valores: funciona como una semilla
aleatorio :: Int -> Int -> Int -> [[Double]]
aleatorio _ 0 _ = [] 
aleatorio n l m = (take n $ randoms (mkStdGen m) :: [Double]):(aleatorio n (l-1) (m + 1))

-- Función que obtiene la transpuesta de una matriz, en donde la matriz es 
-- representada por medio de una lista de listas
-- l: lista de listas que simula una matriz bidimensional
transpuesta :: [[Double]] -> [[Double]]
transpuesta []        = []
transpuesta l  
   | auxtrans l == [] = (transpuesta (auxtrans2 l))
   | otherwise        = (auxtrans l):(transpuesta (auxtrans2 l))

-- Función auxiliar para la función 'transpuesta', que construye una nuea lista 
-- tomando la cabeza de cada elemento de la lista de listas 
auxtrans :: [[Double]] -> [Double]
auxtrans []     = []
auxtrans (x:xs) 
   | x == []    = []
   |otherwise   = (head x):(auxtrans xs)

-- Función auxiliar para la función 'transpuesta', que quita la cabeza 
-- de cada elemento de la lista de listas.
auxtrans2 :: [[Double]] -> [[Double]]
auxtrans2  []    = []
auxtrans2 (x:xs) 
   | x == []     = []
   |otherwise    = (tail x):(auxtrans2 xs)


-- Función que agrega un 1 como cabeza de una lista, simulando la agregación 
-- del sesgo a cada capa donde es necesario.
sesgo2 :: [Double] -> [Double]
sesgo2 xs = [1] ++ xs

-- Función que agrega un 1 como cabeza a cada elemento de una lista de listas, 
-- simulando la agregación del sesgo a cada capa donde es necesario
sesgo :: [[Double]] -> [[Double]]
sesgo xs = map (\x ->([1] ++ x)) xs


-- Función que simula el producto punto entre matrices, representado a cada matriz 
-- con una lista de listas. 
sumaPesosEntrada :: [[Double]] -> [[Double]] -> [[Double]]
sumaPesosEntrada [] _ = []
sumaPesosEntrada _ [] = []
sumaPesosEntrada (w:ws) xs = (pesosEntrada w xs):(sumaPesosEntrada ws xs)

-- Función que implementa el producto punto entre un vector y una matriz, donde el vector 
-- es una lista y la matriz es una lista de listas.
pesosEntrada :: [Double] -> [[Double]] -> [Double]
pesosEntrada w xs = map (\x -> (foldl (+) 0 (multiplicacion w x))) xs

-- Función que realiza la multiplicación entre dos listas elemento a elemento.
-- Esta función nos ayudará a implementar el producto de matrices.
multiplicacion :: [Double] -> [Double] -> [Double]
multiplicacion [] _ = []
multiplicacion _  [] = []
multiplicacion (x:xs) (y:ys) = x*y:(multiplicacion xs ys)


-- Función que aplica la función de transferencia o activación a cada elemento 
-- de la lista, en nuesro caso usamos la función logistica.
activacion :: [Double] -> [Double]
activacion xs = map (\v -> (1 /(1 + (exp (- v))))) xs

-- Se define la función Softmax para una sola entrada representada por una lista,
-- y devuelve la probabilidad de que ocurra el evento para cada entrada de la lista.
softmax :: [Double] -> [Double]
softmax xs = map (\x -> x/c) l
             where m = maximo xs
                   l = map (\v -> (exp (v-m))) xs
                   c = foldl (+) 0 l 

-- Definimos la función de activación 
logistica :: Double -> Double
logistica z = 1/(1 + (exp (- z)))

-- Función que define la derivada de la función de activación (lógistica).
derivadaLog :: Double -> Double
derivadaLog z = (logistica z)*(1-(logistica z))

-- Función que obtiene el áximo de una lista.
maximo :: [Double] -> Double
maximo = foldr1 (\x acc -> if x > acc then x else acc)

-- Definimos la función 'Entropía cruzada' que utilizaremos como función de error.
entropiaCruzada :: Double -> [Double] -> Double
entropiaCruzada y xs = foldl (+) 0.0 l
                    where m = fromIntegral (length xs)
                          l = map (\a -> -(y*(log a) + (1-y)*(log (1-a)))/m) xs

-- Función que implementa el 'Feed Forward' para una NN, dada una entrada devuelve 
-- una salida. La salida se obtiene al realizar la multiplicación de las 
-- entradas con la matriz de pesos 'w0', y al resultado se le aplica la función de 
-- activación. Siguiendo el principio de propagación hacia adelante realizamos el 
-- mismo procedimiento con la matriz de pesos 'w1' y el resultado anterior, para que
-- finalicemos aplicando la función 'softmax' que se será nuestra salida final. 
-- En resumen, devuelve los resultados parciales que se obtienen en cada capa.
-- w0 : matriz de pesos para las entradas (capa de entrada)
-- w1 : matriz de pesos para la capa oculta 
feedForward :: [[Double]] -> [[Double]] -> IO [[[Double]]]
feedForward w1 w0 = do e <- cargaBase
                       let entrada = datosEntrada e
                       let a0 = sesgo entrada
                       let z1 = transpuesta (sumaPesosEntrada w0 a0)
                       let a1 = map (\x -> sesgo2 (activacion x)) z1
                       let z2 = transpuesta (sumaPesosEntrada w1 a1)
                       let a2 = map (\z -> softmax z) z2
                       return $ [a2,a1,a0,z2,z1]


-- Función que obtiene los resultados parciales de la capa 'n'
-- n : el índice de la capa
getCapa ::Int -> [[[Double]]] -> [[Double]]
getCapa 1 x = head x
getCapa n (x:xs) = getCapa (n-1) xs

-- Función que obtiene la matriz de pesos 'w0' representada por una lista de listas.
getW0 :: IO [[Double]]
getW0 = do w1 <- carga pesos0
           return $ (convierte w1)

-- Función que obtiene la matriz de pesos 'w1' representada por una lista de listas.
getW1 :: IO [[Double]]
getW1 = do w2 <- carga pesos1 
           return $ (convierte w2)

-- Función que extrae únicamente las salidas deseadas del archivo de entrada,
-- y las devuelve en una lista de tipo 'Double'.
-- NOTA : en el archivo de entrada, cada línea corresponde a un ejemplor junto 
--        con su salida esperada siendo el último número de la línea.  
salida :: IO [Double]
salida = do input <- cargaBase
            let s = datosSalida input
            return $ (salidaAux s)

-- Función que convierte una lista de tipo String en una lista de tipo Double,
-- de manera que limpia cada elemento de la lsta quitando el retorno de carro '\r'
salidaAux :: [String] -> [Double]
salidaAux [] = []
salidaAux (x:xs) = (read (limpia3 x)::Double):(salidaAux xs)

-- Función que quita de un String el retorno de carro '\r', y devuelve el nuevo String.
limpia3 :: String -> String
limpia3 [] = []
limpia3 (x:xs)
  | x == '\r' = limpia3 xs
  | otherwise = x:(limpia3 xs)


-- Función que almacena los datos de entrada leidos desde un archivo en una lista de 
-- listas de tipo String. Cada línea del archivo es una elemento de la lista. 
cargaBase :: IO [[String]]
cargaBase = do contenido <- readFile baseDatos
               return $ map (splitOn "," ) (splitOn "\n" contenido )

-- Función que obtiene únicamente los ejemplares de los datos de entrada, separando
-- las salidas esperadas de cada ejemplar y almacenando dichos ejemplares en una 
-- lista de listas de tipo Double
-- xs : representa todos los ejemplares almacenados en una lista de listas de tipo String 
datosEntrada :: [[String]] -> [[Double]]
datosEntrada xs = map (entradaAux 10) xs

-- Función que obtiene únicamente las salidas esperadas de cada ejemplar.
-- xs : representa todos los ejemplares almacenados en una lista de listas de tipo String 
datosSalida :: [[String]] -> [String]
datosSalida xs = map (\x -> (head (reverse x))) xs

-- Función que obtiene los primero 'n' elementos de una lista de tipo String y 
-- los devuelve como elementos de una lista de tipo Double. 
-- n : número de elemntos que consideraran
entradaAux :: Int -> [String] -> [Double]
entradaAux 0 _ = []
entradaAux _ [] = []
entradaAux n (x:xs) = (read x::Double):(entradaAux (n-1) xs)

-- Función que alamcena una matriz de pesos que se encuentra en un archivo de entrada en 
-- una lista de listas de tipo String, en donde cada línea del archivo es un elemento de 
-- la lista 
carga :: String -> IO [[String]]
carga datos = do w0 <- readFile datos
                 let w = limpia2 (limpia w0)
                 return $ map (splitOn ",") (splitOn "]," w)


-- Función que quita de un String el caracter '['
limpia :: String -> String
limpia [] = []
limpia (x:xs)
   | x == '[' = limpia xs
   | otherwise = x:(limpia xs)

-- Función que quita de un String los dos últimos caracteres
limpia2 :: String -> String
limpia2 l =  reverse (tail (tail (reverse l)))   

-- Función que convierte una lista de lista de tipo String a tipo Double
convierte :: [[String]] -> [[Double]]
convierte xs = map (\x -> (map (\y -> (read y::Double)) x)) xs 

-- Aplica una función a cada uno de los elementos de una lista de listas.
mmap :: (Double -> Double) -> [[Double]] -> [[Double]]
mmap f xs = map (\x -> map f x) xs

-- Función realiza la resta entre un Double y una lista de tipo Double,
-- devolviendo una nueva lista en donde el elemento i-ésimo es la resta  
-- del número dado menos el i-ésimo elemento de la lista de entrada
delta2Aux :: Double -> [Double] -> [Double]
delta2Aux y xs = [-(y-x)| x <- xs]

-- Función que realiza la resta del elemento i-ésimo de una lista de tipo 
-- Double menos el elemento i-ésimo de una lista de lista de tipo Double, 
-- haciendo uso de la función 'delta2Aux' 
delta2 :: [Double] -> [[Double]] -> [[Double]]
delta2 _ [] = []
delta2 [] _ = []
delta2 (y:ys) (x:xs) = (delta2Aux y x):(delta2 ys xs)

-- Función que multiplica dos listas de listas de tipo Double elemento a elemento.
mult :: [[Double]] -> [[Double]] -> [[Double]]
mult _ [] = []
mult [] _ = []
mult (x:xs) (y:ys) = (multiplicacion x y):(mult xs ys)

-- Función que nos permite actualizar las matrices de pesos, usando el método 
-- Back Propagation que como resultado obtenemos los gradientes almacenados en 
-- matrices cuyas dimensiones son iguales a la matriz de peso correspondiente.
-- NOTA : una matriz se representa por una lista de listas
backPropagate :: [[Double]] -> [[Double]] -> IO [[[Double]]]
backPropagate w1 w0 = do y <- salida
                         ys <- feedForward w1 w0 
                         let a2 = getCapa 1 ys
                         let a1 = getCapa 2 ys
                         let a0 = getCapa 3 ys
                         let z2 = getCapa 4 ys
                         let z1 = getCapa 5 ys
                         let m  = fromIntegral (length (head a0)) - 1
                         let d2 = delta2 y a2
                         let g1 = mmap (*(1/m)) (sumaPesosEntrada (transpuesta d2) (transpuesta a1))
                         let derivada = mmap (derivadaLog) z1
                         let d1 = mult (transpuesta (sumaPesosEntrada (tail (transpuesta w1)) d2)) derivada
                         let g0 = mmap (*(1/m)) (sumaPesosEntrada d1 (transpuesta a0))
                         return $ [g1,g0] 

-- Función que realiza la resta entre dos matrices elemento a elemento.
-- NOTA : una matriz se representa por una lista de listas
resta :: [[Double]] -> [[Double]] -> [[Double]]
resta _ [] = []
resta [] _ = []
resta (x:xs) (y:ys) = (restaAux x y):(resta xs ys)

-- Función que realiza la resta entre dos listas de tipo Double elemento a elemento,
-- es decir, se restan los elementos de la posición i-ésima de cada lista.
restaAux :: [Double] -> [Double] -> [Double]
restaAux [] _ = []
restaAux _ [] = []
restaAux (x:xs) (y:ys) = (x-y):(restaAux xs ys)

-- Función que obtiene la matriz con los gradientes asociados a cada
-- matriz de pesos.
-- NOTA : una matriz se representa por una lista de listas
getGradient :: Int -> [[[Double]]] -> [[Double]]
getGradient 1 x = head x
getGradient n (x:xs) = getGradient (n-1) xs

-- Función que implementa el método 'Descenso por el gradiente' sobre
-- una NN. Evalua y ajusta los pesos de la red neuronal de acuerdo a 
-- los datos de entrada y los resultados esperados.
-- Las entradas son los corresponden a los gradientes calculados con 
-- función 'backPropagate'. 
gradientDescent :: [[Double]] -> [[Double]] -> IO [[[Double]]]
gradientDescent w1 w0 = do bp <- backPropagate w1 w0
                           let g1 = getGradient 1 bp
                           let g0 = getGradient 2 bp 
                           let alpha = 0.1
                           let p1 = resta w1 (mmap (*alpha) g1)
                           let p0 = resta w0 (mmap (*alpha) g0)
                           return $ [p1,p0]

-- Función que implementa el entrenamienamineto de nuestra NN (red neuronal).
-- n : comienza su valor en cero, funciona como un contador.
-- f : representa los ciclos a realizar en el entrenamiento.
-- p : matriz de pesos de la capa inicial.
-- w : matriz de pesos de la capa oculta.
trainAux :: Int -> Int -> [[Double]] -> [[Double]] -> IO ()
trainAux n f w p = if (n == 0) then do gd <- gradientDescent w p 
                                       let p1 = getGradient 1 gd 
                                       let p0 = getGradient 2 gd 
                                       training (n + 1) f p1 p0 
                               else do gd <- gradientDescent w p 
                                       let p1 = getGradient 1 gd 
                                       let p0 = getGradient 2 gd
                                       training (n + 1) f p1 p0 

-- Función que inicializa y actualiza los argumentos de la NN necesarios 
-- para llevar a cabo el entrenamiento.
training :: Int -> Int -> [[Double]] -> [[Double]] -> IO ()
training n f w1 w0 = if (n == f) then do writeFile "pesosw1final.data" (show w1)
                                         writeFile "pesosw0final.data" (show w0)
                                 else trainAux n f w1 w0

-- Función obtiene las posiciones de los elementos máximos de cada elemento de 
-- una lista de listas, y las devuelve en una lista de tipo Double.
-- xs : lista de listas que representa todos los ejemplares
predictAux :: [[Double]] -> [Double]
predictAux xs  = map (\x -> (mano 0 x)) xs

-- Función que obtiene la posición del elemento máximo de una lista.
-- NOTA : cada posición de la lista esta asociada a una mano de poker.
-- n : número que representa la posición de la lista.
mano :: Double -> [Double] -> Double
mano n []    = n
mano n (x:xs) = if (x == (maximo (x:xs))) then n else mano (n + 1) xs 

-- Función que regresa el número de aciertos entre las salidas calculadas
-- y la salidas esperadas
accuracy :: [Double] -> [Double] -> Double
accuracy [] _ = 0
accuracy _ [] = 0
accuracy (x:xs) (y:ys) = if (x == y) then 1 + (accuracy xs ys) else accuracy xs ys

-- Función que calcula el nnúmero de aciertos entre el número total de ejemplares.
predict :: IO Double
predict = do w1 <- getW1
             w0 <- getW0
             f <- feedForward w1 w0
             let a2 = getCapa 1 f 
             y <- salida
             let ys = predictAux a2
             let acc = accuracy y ys
             let m = fromIntegral (length ys)
             return $ (acc/m)*100  

