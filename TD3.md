

# Fonctions à 2 variables; représentation graphique

## Exercice 1 : tracés de fonctions

Tracer les fonctions mathématiques suivantes à l'aide des méthodes `imshow` et `plot_surface`. On cherchera à bien nommer les axes, et à rajouter une barre de couleur latéral.

$$ f(x,y) = \sin{\sqrt{x^2+y^2}} $$
pour $x\in [-5,5]$ et $ y\in [-5,5]$, puis $x\in [-50,50]$ et $ y\in [-50,50]$.

```python
#Importation des modules

import numpy as np
import matplotlib.pyplot as plt


# Definition de la fonction f(x,y)

def f(x,y):
    return np.sin(np.sqrt(x**2+y**2))

# Données :

x1 = np.linspace(-5,5,11)
y1 = x1
x2 = 10*x1
y2 = 10*x1


#Corps (1er ensemble)

fig = plt.figure()
xx, yy = np.meshgrid(x1,y1)
z = f(xx,yy)
im = plt.imshow(z, extent=[-5,5,-5,5] )
plt.xlabel('x')
plt.ylabel('y')
c = plt.colorbar(im)
c.set_label('z')
plt.title('Fonction f sur le premier intervalle')
plt.show()

fig1 = plt.figure()
ax=plt.axes(projection='3d')
xx, yy = np.meshgrid(x1,y1)
z = f(xx,yy)
surface1=ax.plot_surface(xx,yy,z)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')
plt.title('Fonction f en représentation 3D')
plt.show()



#Corps (2eme ensemble)

xx, yy = np.meshgrid(x2,y2)
z = f(xx,yy)
fig = plt.figure()
im = plt.imshow(z, extent=[-50,50,-50,50] )
plt.xlabel('x')
plt.ylabel('y')
c = plt.colorbar(im)
c.set_label('z')
plt.title('Fonction f sur le second intervalle')
plt.show()
    
fig1 = plt.figure()
ax=plt.axes(projection='3d')
xx, yy = np.meshgrid(x2,y2)
z = f(xx,yy)
surface1=ax.plot_surface(xx,yy,z)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')
plt.title('Fonction f en représentation 3D')
plt.show()
```

```python
L'image ressemble beaucoup à une oeuvre décorative, cela est plaisant. La seconde car ayant une plus grande résolution.
```

```python

#LA REPONSE ICI
```



- A quoi vous fait penser cette fonction et quelle est la meilleure représentation graphique ?


LA REPONSE ICI (double-clique pour editer la cellule)


- Tracez la fonction suivante avec de nouveau `imshow` et `plot_surface`:


$$ f(x,y) = x^2 - y^2 \text{ pour } x\in [-1,1] \text{ et } y\in [-1,1]$$

```python

#Définition de la fonction f

def f(x,y):
    return (x**2)-(y**2)

#Données

x = np.linspace(-1,1,51)
y = x
# Tracé avec implot 

xx, yy = np.meshgrid(x,y)
z = f(xx,yy)
fig = plt.figure()
im = plt.imshow(z, extent=[-1,1,-1,1] )
plt.xlabel('x')
plt.ylabel('y')
c = plt.colorbar(im)
c.set_label('z')
plt.title('Fonction f')
plt.show()


#Tracé avec plot_surface

fig1 = plt.figure()
ax=plt.axes(projection='3d')
xx, yy = np.meshgrid(x,y)
z = f(xx,yy)
surface1=ax.plot_surface(xx,yy,z)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')
plt.title('Fonction f en représentation 3D')
plt.show()
```

```python

Elle est lisse et on peut visualiser un point selle de la fonction f (0,0)? .
```


- Qu'est-ce qui est remarquable sur cette fonction ?



LA REPONSE ICI (double-clique pour editer la cellule)


## Exercice 2 : analyse d'une fonction 2D

Mathématiquement, la dérivée d'une fonction $f(x,y)$ soit par rapport à la variable $x$, soit par rapport à la variable $y$, est définie par ses dérivées partielles:

$$ \frac{\partial f}{\partial x}(x,y) = \lim\limits_{\epsilon \rightarrow 0} \frac{f(x+\epsilon,y)-f(x,y)}{\epsilon} $$

$$ \frac{\partial f}{\partial y}(x,y) = \lim\limits_{\epsilon \rightarrow 0} \frac{f(x,y+\epsilon)-f(x,y)}{\epsilon} $$

Le gradient d'une fonction est le vecteur des dérivées partielles :

$$\vec{\text{grad}}\ f(x,y) = \vec{\nabla} f(x,y)  = \left(\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y} \right)$$

Numériquement, comme la fonction est discrétisée en $\left\lbrace x_i\right\rbrace$ points, on peut essayer d'approcher cette définition du mieux possible en calculant le taux d'accroissement de la fonction entre deux points les plus proches possibles:

$$\frac{\partial f}{\partial x}(x_i,y_j) = \frac{f(x_{i+1},y_j)-f(x_i,y_j)}{x_{i+1}-x_{i}} $$
$$\frac{\partial f}{\partial y}(x_i,y_j) = \frac{f(x_i,y_{j+1})-f(x_i,y_j)}{y_{j+1}-y_{j}} $$


### 2.1 Gradient

- Ecrire une fonction `grad(f,x,y)` qui calcule les dérivées partielles d'une fonction `f` à deux variables. Les entrées `x` et `y` sont des tableaux 1D, et la sortie un vecteur contenant les dérivées partielles selon $x$ et $y$. On pourra consulter la documentation de la méthode [`np.gradient`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.gradient.html)
en faisant attention à la normalisation de la dérivée.

**Bon à savoir** : pour une fonction 2D, `np.gradient` renvoie la dérivée selon `y` en premier, puis celle selon `x`, car le gradient est d'abord calculé selon les lignes (valeurs de `y`) puis selon les colonnes (valeurs de `x`).

```python

# Importation du module sympy qui sert au calcul différentiel

from sympy import sympify, lambdify, diff, symbols

def grad(f,x,y):
    sympify(f)
    x, y = symbols('x y')
    dfdx = lambdify((x,y),diff(f,x))
    dfdy = lambdify((x,y),diff(f,y))
    c = lambda x,y: dfdx(x,y)+dfdy(x,y)
    return c
```


- Tester cette fonction sur $f(x,y)=x^2-y^2$ pour $x\in [-1,1] \text{ et } y\in [-1,1]$. Tracer le résultat.

```python

#Tracé du gradient

xx, yy = np.meshgrid(x,y)
z = grad('x**2-y**2',x,y)(xx,yy)
fig = plt.figure()
im = plt.imshow(z, extent=[-1,1,-1,1] )
plt.xlabel('x')
plt.ylabel('y')
c = plt.colorbar(im)
c.set_label('f\'(x,y)')
plt.title('Variations de la fonction f ')
plt.show()
```


- Est-ce que ça vous semble conforme à ce que donne un calcul à la main ?


LA REPONSE ICI (double-clique pour editer la cellule)


 - Existe-il un point où les dérivées partielles s'annulent toutes les deux ? Pour cela tracer la norme du gradient de $f$, puis à l'aide de la méthode `np.where` rechercher les points $(x,y)$ tels que la norme du gradient est inférieur à un seuil proche de zéro.

```python

# De manière symbolique nous pouvons determiner l'expression des der. part. de f

x, y = symbols('x y')
print(diff(x**2-y**2,x))
print(diff(x**2-y**2,y))
#On retrouve le meme resultat à la main

# Cherchons les points (x,y) tels que le gradient soit compris entre un intervalle proche de 0

extremum = []
x = np.linspace(-1,1,51)
y =x
for i in range(len(x)):
    for j in range(len(y)):
        if -0.0001 < z[i][j] < 0.0001:
            extremum.append((i,j))
extremum

# On remarque que le gradient s'annulle quand x=y.
```

### 2.2 Hessien (facultatif)

- On cherche maintenant à calculer le Hessien de la fonction $f$, définie par:

$$\displaystyle{\nabla^2\ f(x,y) = \begin{pmatrix}
\dfrac{\partial^2 f}{\partial x^2} & \dfrac{\partial^2 f}{\partial x\partial y} \\
\dfrac{\partial^2 f}{\partial y \partial x} & \dfrac{\partial^2 f}{\partial y^2} \\
\end{pmatrix}}$$

Créer une fonction `hessien(f,x,y)` qui a les mêmes entrées que la fonction `grad` mais renvoie la matrice hessienne.

```python

#LA REPONSE ICI
```


- Comparer le résultat donné par la fonction `hessien` à un calcul à la main pour la fonction $f(x,y)$. Que constatez-vous ?

```python
#LA REPONSE ICI
```


- Le format du tableau en sortie de la fonction `hessien` n'est pas pratique à manipuler car c'est une matrice $2\times 2$ contenant des tableaux 1D (s'en convaincre avec la méthode `.shape`). Transposer le résultat pour avoir un tableau à deux entrées contenant des matrices $2\times 2$.

```python

#LA REPONSE ICI
```


- La fonction $f(x,y)$ possède un point selle s'il existe un point $(x,y)$ pour lequel le gradient est nul et le hessien possède un déterminant négatif. Est-ce le cas ?

```python

#LA REPONSE ICI
```


## Exercice 3 : ajustement de fonctions 2D

Un faisceau laser est incident sur une caméra numérique (on suppose que la caméra est protégée par des filtres!). Sur l'image enregistrée on observe une tache lumineuse. 

On souhaiterait extraire automatiquement de cette image les positions horizontale $x_0$ et verticale $y_0$ du centre de la tache lumineuse ainsi que sa largeur $w_x$ et $w_y$ suivant les directions horizontales et verticales.

### 3.1 Chargement de l'image à étudier

L'image `image_laser.tif` peut être ouverte par Python à l'aide du script suivant. Une fois ouverte, elle est stockée dans la mémoire de l'ordinateur sous la forme d'un tableau 2D de réels appelé ici `image1`. 

Le nombre de lignes ($i=0,1...,N_x$) et de colonnes ($j=0,1...,N_y$) dans ce tableau correspond aux nombres de pixels dans les directions horizontale et verticale de l'image. La valeur $I(i,j)$ contenue dans la case $(i,j)$ du tableau est un nombre proportionnel au nombre de photons reçu par le pixel $(i,j)$ de la caméra: on l'appelera "éclairement du pixel".

- Ouvrir puis afficher l'image enregistrée par la caméra. Quelles sont les dimensions (en pixels) de l'image ?

```python

import matplotlib.image as img
import matplotlib.pyplot as plt

fname1='data/image_laser.tif'
image1=img.imread(fname1,format='tif')

# La dimension de l'image est de (50,40)
```

### 3.2 Fonction modèle
Pour extraire automatiquement les positions  horizontale ($x_0$) et verticale ($y_0$) du centre de la tache lumineuse ainsi que ses largeurs horizontale et verticales $w_x$ et $w_y$ on va modéliser la tache lumineuse par une fonction gaussienne à deux dimensions. 

On assimile l'image à trois séries de données: les abscisses $x_{ij}$ et les ordonnées $y_{ij}$ des pixels et leurs éclairements $I_{ij}$. On va modéliser ces séries par un modèle de la forme $I=f(x,y)$ en utilisant une régression non-linéaire des moindres carrés. Les paramètres de la fonction $f$ obtenus par ajustement moindre carré permettront d'en déduire $x_0$, $y_0$, $w_x$ et $w_y$.      

La tache lumineuse sera modélisée par une fonction gaussienne $I=f(x,y)$ de la forme suivante:

$$I=f(x,y)=A_0 \mathrm{exp}\left(-\frac{(x-x_0)^2}{w_x^2}-\frac{(y-y_0)^2}{w_y^2}\right)+e_0$$

- Créer une fonction `gaussienne` qui calcule la fonction $f(x,y)$ précédente.  
- Représenter graphiquement $I=f(x,y)$ en fonction de x et y pour un ensemble de paramètres $x_0$, $y_0$, $w_x$, $w_y$, $A$ et $e_0$ choisis arbitrairement. 


On pourra s'inspirer de l'exemple ci-dessous (quelles sont les dimensions de X, Y et Z, et que représentent-ils ?).

```python

import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt

def fonction_modele(x,y, a0,a1,a2):
    res=a0*x**2+a1*y**2+a2
    return res

x = np.linspace(0, 200, 201)
y = np.linspace(0, 200, 201)
X,Y = np.meshgrid(x, y)

Z=fonction_modele(X,Y,1.0,-2.0,10.0)
plt.figure(1)
plt.pcolor(X,Y,Z, shading="auto")
plt.colorbar()
plt.show()
```

```python

# Définition de la fonction gaussienne
def gaussienne(x,y,A,x_0,y_0,w_x,w_y,e_0):
    h = ((-(x-x_0)**2)/((w_x**2)))
    i = ((-(y-y_0)**2)/((w_y**2)))
    
    res = A*np.exp(h-i)+e_0
    return res
# Tracé de la fonction gaussienne 

x = np.linspace(0, 50, 51)
y = np.linspace(0, 50, 51)
X,Y = np.meshgrid(x, y)

Z=gaussienne(X,Y,1,0,0,100,100,1)
plt.figure()
plt.pcolor(X,Y,Z, shading="auto")
plt.colorbar()
plt.show()
```


### 3.3 Création d'une image  test

Afin de tester notre procédure d'ajustement moindre carré, nous allons créer, à partir de la fonction précédente, une fonction "bruitée" qui simulera une image expérimentale. 
- Ecrire une fonction `gaussiennebruitee`, qui est la somme de la fonction `gaussienne` et d'un nombre aléatoire tiré au sort pour chaque pixel selon une loi gaussienne d'écart type `sigma_noise`. On choisira  `sigma_noise` de façon à ce que l'image apparaisse visiblement bruitée mais en gardant la tache bien discernable à l'oeil.
- Afficher le résultat obtenu.

```python

def gaussiennebruitee(x,y,A,x_0,y_0,w_x,w_y,e_0):
    return gaussienne(x,y,A,x_0,y_0,w_x,w_y,e_0) + [[np.random.random() for i in range(51)] for j in range(51)]

x = np.linspace(0, 50, 51)
y = np.linspace(0, 50, 51)
X,Y = np.meshgrid(x, y)


Z=gaussiennebruitee(X,Y,1,0,0,100,100,1)
plt.figure()
plt.pcolor(X,Y,Z, shading="auto")
plt.colorbar()
plt.show()
```


### 3.4 Fit de la fonction test

Effectuer un ajustement de l'image bruitée par une fonction `gaussienne`. Pour ajuster une fonction 2D, il faut mimer le plus possible ce que vous savez faire avec les fonctions 1D et `curve_fit`, c'est-à-dire donner les données et les abscisses aplaties au format 1D, ainsi que le résultat de la fonction à ajuster.
- On étudiera l'exemple donné ici : http://clade.pierre.free.fr/EDPIF/lecture_note/html/fit.html#fitting-image . Par exemple, créer une fonction `gaussienne_fit(Mxy, a0, x0, y0, wx, wy, e0)` qui prend en permier argument le tableau `Mxy` des abscisses (même si celui-ci est 2D) puis les paramètres à ajuster. A l'intérieur de cette fonction, le tableau `Mxy` est scindé en deux tableaux `x` et `y` à donner à votre fonction `gaussienne` initiale.
- Choisir des parametres initiaux "raisonnables" ($e_0$ à la valeur minimale de $I(x,y)$, $x_0$, $y_0$ au milieu de l'image, taille de l'image laser d'environ 10 pixels...)
- Donner à `curve_fit` la fonction `gaussienne_fit`, le tableau des abscisses 2D `Mxy`, le tableau de données aplaties (c'est-à-dire de dimension 1) réalisé à partir de `gaussienne_bruitee`, et vos paramètres initiaux. 
- Retrouve t-on les paramètres $x_0$, $y_0$, $w_x$ et $w_y$ avec lesquels on avait calculé l'image de test ? 
A partir de quel ecart-type sur le bruit a-t-on un écart significatif ?

```python

#Importation du module scipy.optimize qui contient curve fit

from scipy.optimize import curve_fit


# Définition de la fonction gaussienne fit 

def gaussienne_fit(X,a0,x0,y0,wx,wy,e0):
    x1,y1 = X
    return gaussienne(x1,y1,a0,x0,y0,wx,wy,e0)


#Données

x = np.linspace(0, 50, 51)
y = np.linspace(0, 50, 51)
z = gaussienne_fit((x,y),1,0,0,100,100,1)
# Fit 
p0 = 1,0,0,100,100,1
popt, pcov = curve_fit(gaussienne_fit,(x,y),z,p0)

print(popt,pcov)
```


### 3.5 Test sur une vraie image

Effectuer un ajustement de l'image expérimentale par la fonction `gaussienne`. 
- Afficher les valeurs de  $x_0$, $y_0$, $w_x$ et $w_y$ obtenues grâce à cet ajustement (ne pas oublier l'incertitude). 
- Afficher en rouge sur l'image les points de coordonnées $(x0,y0)$ ainsi imprimer les coordonnées du point acec ses incertitudes $(x0 \pm dx0,y0 \pm dy0)$. On utilisera la fonction `scatter(vx,vy,color='r',s=50)` qui permet d'afficher les points de coordonnées `(vx[i],vy[i])`.

```python

j = image1

popt, pcov = curve_fit(gaussienne_fit,(x,y),j,p0)
```



