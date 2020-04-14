# Projet Python ESA

<p></p>
Le projet a été réalisé par :
<p></p>
<i> <b> YE Maxime</i> </b>
<p></p>
<i><b>Buvat Antoine</i></b>
<p></p>
<i><b>RAOUL-JOURDE Thibaut</i></b>
<p></p>
<p></p>
<u>Sous la supervision de :</u>
<p></p>
<i><b>M. Joseph BASQUIN</i></b>

## Introduction au projet

Nous avons décidé de travailler sur le sujet B : Analyse de sentiments (positif, négatif, neutre) sur des tweets (analyse de texte en langage naturel). 
<p></p>
L'analyse de texte, le texte mining, est de plus en plus en vogue ces dernières années et son importance continue de grandir au fil du temps.
Avec le passage au numérique de nombreuses enseignes, avec les réseaux sociaux, ou les sites comme Amazon se basant complètement sur le service en ligne, les informations textuelles sont devenues extrêmement simples à récupérer.
<P></p>
Ce type d'analyse se retrouve dans de nombreux domaines.  
Cette nouvelle évolution du traitement de la donnée permet notamment de mieux comprendre les clients ou encore d'aider à la détection de la fraude pour ne citer que ces exemples-là.  
Evidemment une simple analyse de texte, ou de sentiments dans notre projet, ne suffit pas, cette analyse est là pour compléter les modèles de machine learning déjà existants.  
Apporter de l'information supplémentaire pour aider à l'amélioration de nos modèles.  
Avant de continuer et après avoir un petit peu loué les mérites de l'analyse de texte, nous allons définir précisément ce qu'est le text mining.

<p></p>

Le text mining est un ensemble de techniques appartenant au domaine de l’intelligence artificielle qui allie les domaines de la linguistique, de la sémantique et du langage, des statistiques et de l’informatique.  
Ces techniques permettent d’extraire et de recréer de l’information à partir d’un corpus de textes (classification, analyse, tendance, etc.).  
Beaucoup utilisé en marketing, le text mining est une discipline très utilisée dans de nombreux domaines tels que la recherche, les sciences politiques et la communication.  

<p></p>

## Explication détaillée du code

<P></p>

Suite à cette brève explication du text mining nous allons entrer dans le coeur de notre sujet.  
Soit, comment nous avons réussi à mettre en place une analyse de sentiments sur une base de données tweeter.

<p></p>

Après l'installation de certains packages nécessaires à notre code (cf. INSTALL), nous avons lancé la préparation des données.  
La préparation des données est un des points les plus importants de toutes analyses. Des données "sales" ne peuvent que détériorer les résultats futurs.  
"Garbage In Garbage Out" est une expression expliquant parfaitement bien pourquoi nous devons préparer nos données correctement.  

<p></p>

Le nettoyage des données a majoritairement consisté à supprimer des morceaux de code HTML pouvant être présents dans les tweets, supprimer les tags et supprimer les liens.  
Les tags étant présents pour permettre à un autre utilisateur de voir le tweet, ils sont là pour créer une interaction entre utilisateurs, ce ne sont donc pas des termes importants à notre analyse.  
Les liens redirigeant vers une nouvelle conversation/page web, ne sont également d'aucune aide dans une analyse de sentiments.  

<p></p>

Une fois le nettoyage effectué et le fichier importé, nous renseignons le sens de notre target.  
0 = Tweet Négatif  
2 = Tweet Neutre  
4 = Tweet Positif  

<P></p>

Par la suite nous créons une fonction decode_sentiment nous indiquant la nature du tweet en fonction de son score pour notre target, via les informations prédéfinis au-dessus.  

<p></p>

Dans la continuité de notre premier nettoyage de table, nous identifions les mots les plus utilisés (en anglais dans notre analyse) mais qui n'auront aucun impact sur la compréhension du texte.  
Un exemple concret serait les mots tels que "The", "a".

</p>

Après ce nettoyage, nous nous concentrons sur la "racinisation" des tweets.  
L'objectif est de conserver la racine du mot plutôt que le mot lui-même.  
Par exemple : "You are creative" deviendra "Be" "Creative".  
Les mots au pluriel seront ramenés au singulier après la mise en place de cette étape de la préparation des données.  

<p></p>

Une fois de plus nous repassons sur nos données afin de supprimer cette fois tous les caratères spéciaux, que nous aurons indiqué dans la fonction que nous créons.  
<p></p>

Dans un second temps, nous séparons donc nos données en 2 échantillons, un échantillon d'apprentissage nous permettant de tester nos modèles, d'apprendre de nos données afin de sortir les résultats voulus.  
Ainsi qu'un échantillon test, permettant de vérifier nos résultats, voir si nous n'avons pas fait de surapprentissage dans notre précédant échantillon.  
Les échantillons seront séparés de manière assez classique, 70-30.  

<p></p>

Pour la mise en place de notre modèle nlp, nous séparons chaque mot les uns des autres.  
Nous stockons chaque mot dans la liste "mot".  
A l'aide du package Gensim, nous pouvons entrainer des vecteurs de mots qui nous servirons plus tard.  

<p></p>

Nous continuons avec la tokénisation de nos mots.  

Puis, nous ajoutons une nouvelle modalité neutre. De plus, nous transformons avec le labelencoder, les modalités qualitatives en entier numérique.  
Ensuite, nous convertissons les vecteurs de mots en entier et nous ajustons notre modèle.  

<P></p>

Une nouvelle fonction est créée, decode_sentiment, permettant de trouver le sentiment dans le texte en fonction d'un score.  
Score utilisé : (Données pris arbitrairement)  
<p></p>
 =< 0.35 : Sentiment négatif associé au tweet  
 <p></p>
 0.35 < x < 0.65 : Sentiment neutre associé au tweet  
 <P></p>
 >= 0.65 : Sentiment positif associé au tweet  

<p></p>

Nous continuons de créer des fonctions, cette fois-ci c'est la fonction predict qui nous aide à prédire le sentiment d'un texte qui sera renseigné en paramètre.  

<p></p>

Pour simplifier notre code et son utilisation, nous faisons appel aux fichiers contenant les modèles et ainsi éviter de devoir tout relancer à chaque fois.  

<p></p>

Nous utilisons ensuite notre fonction sur la base test.  
Nous créons donc un dataframe vide qui nous servira par la suite de table de résultats.  
Après avoir importé notre échantillon test, nous réalisons une boucle qui nous aide à appliquer notre fonction sur chacun des tweets de notre table.  

<p></p>

Afin d'obtenir un vecteur de mot comme lors de nos prévisions, nous transformons notre variable target.  

<p></p>

Puis pour finir nous calculons le pourcentage de bonne classification.
<p></p>
&nbsp;

## Ressenti lors du projet

Un des points importants qu'il fallait réussir à bien prendre en compte comme expliqué avant dans notre code, c'est de bien rassembler les mots qui vont ensemble.
Effectivement entre "aimer" et "ne pas aimer", il y a une énorme différence et il faut que l'analyseur puisse la comprendre.
Voir le mot "aimer" ne devrait pas suffire pour dire que le tweet est positif, ce qui entoure le mot doit pouvoir être pris en compte.
C'est sûrement un des points les plus importants de l'analyse de sentiments.

<p></p>

## Pistes d'évolutions

Une piste d'évolution, du moins à notre niveau serait de gérer des émotions plus complexes que les simples sentiments aimer, ne pas aimer, positif, négatif.  
Il faudrait pouvoir aller au-delà du résultat binaire et faire une échelle de sentiments.  
Découvrir si le sentiment négatif, est un sentiment qui est tellement proche de la haine qu'il est impossible de faire changer cette personne d'avis ou si au contraire c'est un sentiment assez faible et qu'avec un simple changement on pourrait le faire basculer vers le positif.  
C'est quelque chose qui pourrait être extrêmement utile pour les entreprises qui veulent s'investir auprès de leurs clients.   
Gérer l'ironie ou le sarcasme est également un problème, il faudrait presque pouvoir lire entre les lignes, c'est seulement possible après un long entraiment de modèle de machine learning.  
Les différences de langues peuvent également être un problème qu'il faudra régler dans le futur.  
La langue française contient de nombreuses subtilités qui ne sont pas forcément visibles par la machine pour le moment.  
Les langues des pays asiatiques sont également un problème, il faut pouvoir comprendre les signes et non plus des lettres.  
En japonais, un même symbole peut avoir plusieurs significations en fonction du type de kanji par exemple.  

<p></p>

Ce sont pour nous les premières pistes qu'il faudrait explorer.

<p></p>


#### Lien utilie à la compréhension du code

Stopword et Racinisation  
https://openclassrooms.com/fr/courses/4470541-analysez-vos-donnees-textuelles/4854971-nettoyez-et-normalisez-les-donnees

<p></p>

Package/ Fonction Gensim (pour entrainer les vecteurs de mots)  
https://radimrehurek.com/gensim/models/word2vec.html



## Astuce 

Afin d'éviter de devoir relancer tout le code (ce qui peut être extrêment long), vous pouvez directement télécharger nos résulats et simplement lancer à partir de la 3ème partie.  
Vous trouverez les fichiers model et tokenizer au lien suivant: 

https://wetransfer.com/downloads/65c608fac1cc4a761dbcecb6c84cc19620200414142433/69b020?fbclid=IwAR0akUvQ5CimurfMZHT6iZ7AQ4CWhBoOTrljtzbDFYqbLiP4XV18Iu7QFkQ

Si le lien est périmé, vous pouvez nous contacter à l'une des adresses suivantes:

maxime.ye@gmail.com  
antoinebuvat74@gmail.com  
thibaut.raouljourde@gmail.com  

Il est préférable de tout enregistrer au même endroit sur votre ordinateur.

