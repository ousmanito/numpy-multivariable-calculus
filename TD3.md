---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

Merci de **ne pas modifier** le nom de ce notebook (même pour y inclure son nom).

Quelques conseils:
- pour exécutez une cellule, cliquez sur le bouton *Exécuter* ci-dessus ou tapez **Shift+Enter**
- si l'exécution d'une cellule prend trop de temps, sélectionner dans le menu ci-dessus *Noyau/Interrompre*
- en cas de très gros plantage *Noyau/Redémarrer*
- **sauvegardez régulièrement vos réponses** en cliquant sur l'icone disquette ci-dessus à gauche, ou *Fichier/Créer une nouvelle sauvegarde*

----------------------------------------------------------------------------

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "56753727a3fa402ec8e54532381bc845", "grade": false, "grade_id": "cell-17fbea484a8c4552", "locked": true, "schema_version": 3, "solution": false, "task": false}}

# TD 3 : fonctions à 2 variables; représentation graphique

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "61264917847fb47ca76b84a6c995563c", "grade": false, "grade_id": "cell-1c459b930496b4ab", "locked": true, "points": 7, "schema_version": 3, "solution": false, "task": true}}

## Exercice 1 : tracés de fonctions

Tracer les fonctions mathématiques suivantes à l'aide des méthodes `imshow` et `plot_surface`. On cherchera à bien nommer les axes, et à rajouter une barre de couleur latéral.

$$ f(x,y) = \sin{\sqrt{x^2+y^2}} $$
pour $x\in [-5,5]$ et $ y\in [-5,5]$, puis $x\in [-50,50]$ et $ y\in [-50,50]$.

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 8f16373c4e07b15ec11da784bab4d6c4
  grade: false
  grade_id: cell-e21d7eeda35647ca
  locked: false
  schema_version: 3
  solution: true
  task: false
---
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

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: b25ea5babc148f2ba6d1976f321d8cc6
  grade: false
  grade_id: cell-973a685093dbde74
  locked: false
  schema_version: 3
  solution: true
  task: false
---
L'image ressemble beaucoup à une oeuvre décorative, cela est plaisant. La seconde car ayant une plus grande résolution.
```

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 1ee3467e9886294fdd2b24ea25ec189b
  grade: false
  grade_id: cell-aadb74aa134487cf
  locked: false
  schema_version: 3
  solution: true
  task: false
---
#LA REPONSE ICI
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "b62ec9989dc32299ba5b1fc660444fbf", "grade": false, "grade_id": "cell-91966229a3c3e764", "locked": true, "schema_version": 3, "solution": false, "task": false}}

- A quoi vous fait penser cette fonction et quelle est la meilleure représentation graphique ?

+++ {"deletable": false, "nbgrader": {"cell_type": "markdown", "checksum": "7e5f006d584c057be49453cd7c205881", "grade": true, "grade_id": "cell-904e2b7f25f93627", "locked": false, "points": 1, "schema_version": 3, "solution": true, "task": false}}

LA REPONSE ICI (double-clique pour editer la cellule)

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "7d7bd7ec92310c2902041fd495f66534", "grade": false, "grade_id": "cell-a0d040789f96b9b1", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": true}}

- Tracez la fonction suivante avec de nouveau `imshow` et `plot_surface`:
$$ f(x,y) = x^2 - y^2 \text{ pour } x\in [-1,1] \text{ et } y\in [-1,1]$$

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 987ab1891a0f71ae9068b15139eae50e
  grade: false
  grade_id: cell-ee982fc2a6d9f87d
  locked: false
  schema_version: 3
  solution: true
  task: false
---
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

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 9e13d70533a2b4100e8accec040eee6c
  grade: false
  grade_id: cell-f3332d43e73a9534
  locked: false
  schema_version: 3
  solution: true
  task: false
---
Elle est lisse et on peut visualiser un point selle de la fonction f (0,0)? .
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "e357e6810cf48a9d21a695c59dcd4dd9", "grade": false, "grade_id": "cell-6df1d387a1940551", "locked": true, "schema_version": 3, "solution": false, "task": false}}

- Qu'est-ce qui est remarquable sur cette fonction ?

+++ {"deletable": false, "nbgrader": {"cell_type": "markdown", "checksum": "7a795e1d3e3cf589c033ada77d372436", "grade": true, "grade_id": "cell-645ab84307beed7b", "locked": false, "points": 1, "schema_version": 3, "solution": true, "task": false}}

LA REPONSE ICI (double-clique pour editer la cellule)

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "266a10054646a8056b1afd2761591e53", "grade": false, "grade_id": "cell-b603c7e708e50158", "locked": true, "schema_version": 3, "solution": false, "task": false}}

## Exercice 2 : analyse d'une fonction 2D

Mathématiquement, la dérivée d'une fonction $f(x,y)$ soit par rapport à la variable $x$, soit par rapport à la variable $y$, est définie par ses dérivées partielles:

$$ \frac{\partial f}{\partial x}(x,y) = \lim\limits_{\epsilon \rightarrow 0} \frac{f(x+\epsilon,y)-f(x,y)}{\epsilon} $$

$$ \frac{\partial f}{\partial y}(x,y) = \lim\limits_{\epsilon \rightarrow 0} \frac{f(x,y+\epsilon)-f(x,y)}{\epsilon} $$

Le gradient d'une fonction est le vecteur des dérivées partielles :

$$\vec{\text{grad}}\ f(x,y) = \vec{\nabla} f(x,y)  = \left(\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y} \right)$$

Numériquement, comme la fonction est discrétisée en $\left\lbrace x_i\right\rbrace$ points, on peut essayer d'approcher cette définition du mieux possible en calculant le taux d'accroissement de la fonction entre deux points les plus proches possibles:

$$\frac{\partial f}{\partial x}(x_i,y_j) = \frac{f(x_{i+1},y_j)-f(x_i,y_j)}{x_{i+1}-x_{i}} $$
$$\frac{\partial f}{\partial y}(x_i,y_j) = \frac{f(x_i,y_{j+1})-f(x_i,y_j)}{y_{j+1}-y_{j}} $$

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "6f17b04dab6ed77d09865bb3ce4d6aec", "grade": false, "grade_id": "cell-2ced33be94a9e94f", "locked": true, "points": 3, "schema_version": 3, "solution": false, "task": true}}

### 2.1 Gradient

- Ecrire une fonction `grad(f,x,y)` qui calcule les dérivées partielles d'une fonction `f` à deux variables. Les entrées `x` et `y` sont des tableaux 1D, et la sortie un vecteur contenant les dérivées partielles selon $x$ et $y$. On pourra consulter la documentation de la méthode [`np.gradient`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.gradient.html)
en faisant attention à la normalisation de la dérivée.

**Bon à savoir** : pour une fonction 2D, `np.gradient` renvoie la dérivée selon `y` en premier, puis celle selon `x`, car le gradient est d'abord calculé selon les lignes (valeurs de `y`) puis selon les colonnes (valeurs de `x`).

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: c5d4457c10e30a26355c043f53c42ffa
  grade: false
  grade_id: cell-bbe15d8480b7d718
  locked: false
  schema_version: 3
  solution: true
  task: false
---
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

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "247b4aeb8fc0999651033b651985c7f9", "grade": false, "grade_id": "cell-30ae654707e49e8f", "locked": true, "points": 4, "schema_version": 3, "solution": false, "task": true}}

- Tester cette fonction sur $f(x,y)=x^2-y^2$ pour $x\in [-1,1] \text{ et } y\in [-1,1]$. Tracer le résultat.

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 33476b8648c7fa355f311ce32ee0a80b
  grade: false
  grade_id: cell-d0a06ba6f2269624
  locked: false
  schema_version: 3
  solution: true
  task: false
---
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

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "28f2c56108fe8e08c751fd7b62cc5b0e", "grade": false, "grade_id": "cell-da0d134d549a373b", "locked": true, "schema_version": 3, "solution": false, "task": false}}

- Est-ce que ça vous semble conforme à ce que donne un calcul à la main ?

+++ {"deletable": false, "nbgrader": {"cell_type": "markdown", "checksum": "33f44166806ec4dcbc88c253fcc6cc71", "grade": true, "grade_id": "cell-4731c0847ccad72f", "locked": false, "points": 2, "schema_version": 3, "solution": true, "task": false}}

LA REPONSE ICI (double-clique pour editer la cellule)

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "1954409e42900a240bd5a7d9045b6790", "grade": false, "grade_id": "cell-ced7bdfac3e8856d", "locked": true, "points": 4, "schema_version": 3, "solution": false, "task": true}}

 - Existe-il un point où les dérivées partielles s'annulent toutes les deux ? Pour cela tracer la norme du gradient de $f$, puis à l'aide de la méthode `np.where` rechercher les points $(x,y)$ tels que la norme du gradient est inférieur à un seuil proche de zéro.

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 7c3963b70dd8aba0267dad5fa48abb84
  grade: false
  grade_id: cell-1cfaf81e1f9a49cc
  locked: false
  schema_version: 3
  solution: true
  task: false
---
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

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: b83dc15ecbe0350bac7981b47f21247c
  grade: false
  grade_id: cell-40e0942947266ca6
  locked: false
  schema_version: 3
  solution: true
  task: false
---
#LA REPONSE ICI
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "1a9d624221be9d1797981643fde173f0", "grade": false, "grade_id": "cell-500e0ca223eaae20", "locked": true, "schema_version": 3, "solution": false, "task": false}}

- Comparer le résultat donné par la fonction `hessien` à un calcul à la main pour la fonction $f(x,y)$. Que constatez-vous ?

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: cacbf38cb24c4323728a8775c1120418
  grade: false
  grade_id: cell-b11de9e85c42d161
  locked: false
  schema_version: 3
  solution: true
  task: false
---
#LA REPONSE ICI
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "a3b11ae82bb9f925075cea9966fb4f27", "grade": false, "grade_id": "cell-be6bcaa9aef1bde3", "locked": true, "schema_version": 3, "solution": false, "task": false}}

- Le format du tableau en sortie de la fonction `hessien` n'est pas pratique à manipuler car c'est une matrice $2\times 2$ contenant des tableaux 1D (s'en convaincre avec la méthode `.shape`). Transposer le résultat pour avoir un tableau à deux entrées contenant des matrices $2\times 2$.

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 58c317a2633612980a0d8e0c6852214e
  grade: false
  grade_id: cell-106f4b00c226038f
  locked: false
  schema_version: 3
  solution: true
  task: false
---
#LA REPONSE ICI
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "33357d0c0ac5ab30501a4a687903a976", "grade": false, "grade_id": "cell-b19a8aebb7969987", "locked": true, "schema_version": 3, "solution": false, "task": false}}

- La fonction $f(x,y)$ possède un point selle s'il existe un point $(x,y)$ pour lequel le gradient est nul et le hessien possède un déterminant négatif. Est-ce le cas ?

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: c645da7a75ab2e6f3d996015858cc264
  grade: false
  grade_id: cell-e8a0df168b6dc9b2
  locked: false
  schema_version: 3
  solution: true
  task: false
---
#LA REPONSE ICI
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "32341226932d2e053d8eaca6573a4de4", "grade": false, "grade_id": "cell-ad691b4083e60e88", "locked": true, "points": 2, "schema_version": 3, "solution": false, "task": true}}

## Exercice 3 : ajustement de fonctions 2D

Un faisceau laser est incident sur une caméra numérique (on suppose que la caméra est protégée par des filtres!). Sur l'image enregistrée on observe une tache lumineuse. 

On souhaiterait extraire automatiquement de cette image les positions horizontale $x_0$ et verticale $y_0$ du centre de la tache lumineuse ainsi que sa largeur $w_x$ et $w_y$ suivant les directions horizontales et verticales.

### 3.1 Chargement de l'image à étudier

L'image `image_laser.tif` peut être ouverte par Python à l'aide du script suivant. Une fois ouverte, elle est stockée dans la mémoire de l'ordinateur sous la forme d'un tableau 2D de réels appelé ici `image1`. 

Le nombre de lignes ($i=0,1...,N_x$) et de colonnes ($j=0,1...,N_y$) dans ce tableau correspond aux nombres de pixels dans les directions horizontale et verticale de l'image. La valeur $I(i,j)$ contenue dans la case $(i,j)$ du tableau est un nombre proportionnel au nombre de photons reçu par le pixel $(i,j)$ de la caméra: on l'appelera "éclairement du pixel".

- Ouvrir puis afficher l'image enregistrée par la caméra. Quelles sont les dimensions (en pixels) de l'image ?

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 2727f6cac1b7fe90441526686f373a3b
  grade: false
  grade_id: cell-d5898634d19d6548
  locked: false
  schema_version: 3
  solution: true
  task: false
---
import matplotlib.image as img
import matplotlib.pyplot as plt

fname1='data/image_laser.tif'
image1=img.imread(fname1,format='tif')

# La dimension de l'image est de (50,40)
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "a45cc5b1dd8dcf60dbe8a0f3e284eea5", "grade": false, "grade_id": "cell-8f66c96237aee0c0", "locked": true, "points": 0, "schema_version": 3, "solution": false, "task": true}}

### 3.2 Fonction modèle
Pour extraire automatiquement les positions  horizontale ($x_0$) et verticale ($y_0$) du centre de la tache lumineuse ainsi que ses largeurs horizontale et verticales $w_x$ et $w_y$ on va modéliser la tache lumineuse par une fonction gaussienne à deux dimensions. 

On assimile l'image à trois séries de données: les abscisses $x_{ij}$ et les ordonnées $y_{ij}$ des pixels et leurs éclairements $I_{ij}$. On va modéliser ces séries par un modèle de la forme $I=f(x,y)$ en utilisant une régression non-linéaire des moindres carrés. Les paramètres de la fonction $f$ obtenus par ajustement moindre carré permettront d'en déduire $x_0$, $y_0$, $w_x$ et $w_y$.      

La tache lumineuse sera modélisée par une fonction gaussienne $I=f(x,y)$ de la forme suivante:

$$I=f(x,y)=A_0 \mathrm{exp}\left(-\frac{(x-x_0)^2}{w_x^2}-\frac{(y-y_0)^2}{w_y^2}\right)+e_0$$

- Créer une fonction `gaussienne` qui calcule la fonction $f(x,y)$ précédente.  
- Représenter graphiquement $I=f(x,y)$ en fonction de x et y pour un ensemble de paramètres $x_0$, $y_0$, $w_x$, $w_y$, $A$ et $e_0$ choisis arbitrairement. 


On pourra s'inspirer de l'exemple ci-dessous (quelles sont les dimensions de X, Y et Z, et que représentent-ils ?).

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 9f9fb89229fae3d0d38e8d6e5ce5b464
  grade: false
  grade_id: cell-a197f225986e3a55
  locked: true
  schema_version: 3
  solution: false
  task: false
---
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

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 2022d4e2d13d5129c74e4719d7929b73
  grade: false
  grade_id: cell-dfe9d4c5d341405a
  locked: false
  schema_version: 3
  solution: true
  task: false
---
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

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "01225714d696df354ffb6c28041a929b", "grade": false, "grade_id": "cell-3d8f4a0d17712ff5", "locked": true, "points": 3, "schema_version": 3, "solution": false, "task": true}}

### 3.3 Création d'une image  test

Afin de tester notre procédure d'ajustement moindre carré, nous allons créer, à partir de la fonction précédente, une fonction "bruitée" qui simulera une image expérimentale. 
- Ecrire une fonction `gaussiennebruitee`, qui est la somme de la fonction `gaussienne` et d'un nombre aléatoire tiré au sort pour chaque pixel selon une loi gaussienne d'écart type `sigma_noise`. On choisira  `sigma_noise` de façon à ce que l'image apparaisse visiblement bruitée mais en gardant la tache bien discernable à l'oeil.
- Afficher le résultat obtenu.

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 46d03d2a9109084da139da3ce076449a
  grade: false
  grade_id: cell-8e43e74300ee8418
  locked: false
  schema_version: 3
  solution: true
  task: false
---
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

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "37be6814d5bd8d933490854d46111117", "grade": false, "grade_id": "cell-c1a7294e422a99de", "locked": true, "points": 6, "schema_version": 3, "solution": false, "task": true}}

### 3.4 Fit de la fonction test

Effectuer un ajustement de l'image bruitée par une fonction `gaussienne`. Pour ajuster une fonction 2D, il faut mimer le plus possible ce que vous savez faire avec les fonctions 1D et `curve_fit`, c'est-à-dire donner les données et les abscisses aplaties au format 1D, ainsi que le résultat de la fonction à ajuster.
- On étudiera l'exemple donné ici : http://clade.pierre.free.fr/EDPIF/lecture_note/html/fit.html#fitting-image . Par exemple, créer une fonction `gaussienne_fit(Mxy, a0, x0, y0, wx, wy, e0)` qui prend en permier argument le tableau `Mxy` des abscisses (même si celui-ci est 2D) puis les paramètres à ajuster. A l'intérieur de cette fonction, le tableau `Mxy` est scindé en deux tableaux `x` et `y` à donner à votre fonction `gaussienne` initiale.
- Choisir des parametres initiaux "raisonnables" ($e_0$ à la valeur minimale de $I(x,y)$, $x_0$, $y_0$ au milieu de l'image, taille de l'image laser d'environ 10 pixels...)
- Donner à `curve_fit` la fonction `gaussienne_fit`, le tableau des abscisses 2D `Mxy`, le tableau de données aplaties (c'est-à-dire de dimension 1) réalisé à partir de `gaussienne_bruitee`, et vos paramètres initiaux. 
- Retrouve t-on les paramètres $x_0$, $y_0$, $w_x$ et $w_y$ avec lesquels on avait calculé l'image de test ? 
A partir de quel ecart-type sur le bruit a-t-on un écart significatif ?

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 01b7dfebe65628fdc34b426cbb01c89a
  grade: false
  grade_id: cell-0a9f56d8afd0078d
  locked: false
  schema_version: 3
  solution: true
  task: false
---
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

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "3a6a7a98502a0bf41840975fb1936e47", "grade": false, "grade_id": "cell-8cc33f091bf7f689", "locked": true, "points": 3, "schema_version": 3, "solution": false, "task": true}}

### 3.5 Test sur une vraie image

Effectuer un ajustement de l'image expérimentale par la fonction `gaussienne`. 
- Afficher les valeurs de  $x_0$, $y_0$, $w_x$ et $w_y$ obtenues grâce à cet ajustement (ne pas oublier l'incertitude). 
- Afficher en rouge sur l'image les points de coordonnées $(x0,y0)$ ainsi imprimer les coordonnées du point acec ses incertitudes $(x0 \pm dx0,y0 \pm dy0)$. On utilisera la fonction `scatter(vx,vy,color='r',s=50)` qui permet d'afficher les points de coordonnées `(vx[i],vy[i])`.

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: f7b952bf28761e81f3da3f4a2007d499
  grade: false
  grade_id: cell-98fa23f4834125a1
  locked: false
  schema_version: 3
  solution: true
  task: false
---
j = image1

popt, pcov = curve_fit(gaussienne_fit,(x,y),j,p0)
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```
