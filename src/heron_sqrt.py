# initialisation
x = 1
erreur = 1
précision = 0.001
# boucle itérative
Tant que erreur > précision:
	# mise à jour de l'estimation
	x = (x + 2/x)/2
	# mise à jour de l'erreur avec la nouvelle valeur de x
	erreur = |x^2 - 2|



la notion d'erreur d'estimation: erreur = |x^2 - 2|
la mise à jour itérative de l'estimation: x = (x + 2/x)/2


Tant que erreur > precision:
	# maj de l'estimation
	x = (x + 2/x)/2
	# mise à jour de l'erreur avec la nouvelle valeur de x
	erreur = sqrt(x^2 - 2)^2

# step 3
using the error to update the x



