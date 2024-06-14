import numpy as np
from math import e
from random import random, randint

vsiNivoji = []
vseUtezi = []
vsiPopravki = []
zacetniNivoji = []
zeljeneVrednosti = []
stopnjaUcenja = 0.1

class nodeLayer:
	def __init__(self,vektor_vrednosti,indeks):
		self.vektor_vrednosti = vektor_vrednosti
		self.indeks = indeks
		self.delte = None

	def izracunajVrednost(self):
		pred = self.vektor_vrednosti

		self.vektor_vrednosti = np.ravel(np.matmul(vseUtezi[self.indeks-1].matrika_vtezi,vsiNivoji[self.indeks-1].vektor_vrednosti))
		

		self.vektor_vrednosti = np.add(self.vektor_vrednosti,vsiPopravki[self.indeks -1].vektor_popravkov)
		
		self.vektor_vrednosti = np.array([sigmoid(xi) for xi in self.vektor_vrednosti])

		#print("sprememba nivoja:")
		#print(pred-self.vektor_vrednosti)

	def izracunajDelte(self, zadnja=False, zeljene=None):
		if self.indeks == 0:
			return

		if zadnja:
			self.delte = np.multiply(np.multiply(np.multiply(np.subtract(zeljene, self.vektor_vrednosti),2), self.vektor_vrednosti),np.subtract(1,self.vektor_vrednosti))
		else:
			self.delte = np.ravel(vseUtezi[self.indeks].matrika_vtezi.T @ vsiNivoji[self.indeks+1].delte) * vsiNivoji[self.indeks].vektor_vrednosti * (1-vsiNivoji[self.indeks].vektor_vrednosti)
			#print(self.delte)


class weightLayer:
	def __init__(self,matrika_vtezi,indeks):
		self.matrika_vtezi = matrika_vtezi
		self.indeks = indeks

	def popraviUtezi(self):
		
		prej = (vseUtezi[self.indeks-1].matrika_vtezi)
		self.matrika_vtezi = self.matrika_vtezi - stopnjaUcenja*np.asmatrix(vsiNivoji[self.indeks].delte).T @ np.asmatrix(vsiNivoji[self.indeks-1].vektor_vrednosti)
		
		if self.indeks == None:
			print("spremembe utezi:")
			print((vseUtezi[self.indeks-1].matrika_vtezi-prej))

class biasLayer:
	def __init__(self,vektor_popravkov,indeks):
		self.vektor_popravkov = vektor_popravkov
		self.indeks = indeks

	def popraviPopravke(self):
		self.vektor_popravkov = self.vektor_popravkov - stopnjaUcenja*vsiNivoji[self.indeks].delte


def shrani(podatek, ime="file.txt"):
	np.savetxt(ime, podatek)

def nalozi(ime="file.txt"):
	return np.loadtxt(ime)

def shraniVse():
	for popravek in vsiPopravki:
		shrani(popravek.vektor_popravkov,"shranjevanje/popravki{0}.txt".format(popravek.indeks))
	for vtezi in vseUtezi:
		shrani(vtezi.matrika_vtezi,"shranjevanje/vtezi{0}.txt".format(vtezi.indeks))

def naloziVse(stNivojev):
	global vsiPopravki
	global vseUtezi
	vseUtezi = []
	vsiPopravki = []
	for i in range(stNivojev):
		vsiPopravki.append(biasLayer(nalozi("shranjevanje/popravki{0}.txt".format(i+1)),i+1))
		vseUtezi.append(weightLayer(nalozi("shranjevanje/vtezi{0}.txt".format(i+1)),i+1))


def sigmoid(stevilo):
	if stevilo> 37:
		return 0.9999999999999999
	if stevilo<-37:
		return 8.533047625744083e-17
	else:
		return 1/(1+e**-stevilo)

def nastaviZacetniNivo(indeks_slike):
	vsiNivoji[0] = nodeLayer(np.asarray(zacetniNivoji[indeks_slike][1:]), 0)
	for row in zacetniNivoji:
		zeljeneVrednosti.append(int(row[0]))



def beriZacetniDokument():
	zacetniNivoji = np.loadtxt("mnist_train_nov.txt")
	return zacetniNivoji



def nastaviOstaleNivoje(nivoji, prva=True):
	if prva:
		for i in range(len(nivoji)-1):
			#vtezi
			m = np.empty((nivoji[i+1],nivoji[i]))

			for row in range(len(m)):
				for col in range(len(m[row])):
					m[row][col] = (random()*2)-1
			vseUtezi.append(weightLayer(m.copy(),i+1))

			#popravki

			v = []
			for col in range(nivoji[i+1]):
				v.append(randint(-10,10))
			vsiPopravki.append(biasLayer(v.copy(),i+1))

	for i in range(len(nivoji)):
		v = []
		for n in range(nivoji[i]):
			v.append(0)
		v = np.asarray(v)

		vsiNivoji.append(nodeLayer(v.copy(),i))


def popraviVse():
	for nivo in vsiNivoji:
		if nivo.indeks != 0:
			nivo.izracunajVrednost()

def popraviDelte(zeljene):
	st_nivojev = len(vsiNivoji)
	for i in range(st_nivojev):
		if i == 0:
			vsiNivoji[st_nivojev-i-1].izracunajDelte(True,zeljene)
		else:
			vsiNivoji[st_nivojev-i-1].izracunajDelte()


def enoUcenje():
	steviloNivojev = len(vsiPopravki)
	for i in range(steviloNivojev):
		vsiPopravki[steviloNivojev-i-1].popraviPopravke()
		vseUtezi[steviloNivojev-i-1].popraviUtezi()

#iz indeksa slike prebere katera vrednost je pravilna in vrne tabelo za racunanje delt
def zeljeniNivojiNastavi(indeks):

	zeljeniNivoji = [0]*10
	zeljeniNivoji[zeljeneVrednosti[indeks]] = 1
	#print(zeljeniNivoji)
	return zeljeniNivoji


def izpisi_stanje():
	for e in vseUtezi:
		print(e.matrika_vtezi)
		print(e.indeks)
	print()
	for e in vsiNivoji:
		print(e.vektor_vrednosti)
		print(e.indeks)

	print()

	for e in vsiPopravki:
		print(e.vektor_popravkov)
		print(e.indeks)



indeksSlike = 3
zacetniNivoji= beriZacetniDokument()
beriZacetniDokument()
nastaviOstaleNivoje([784,16,16,10],False)
naloziVse(3)
nastaviZacetniNivo(0)
pravilne = 0
for j in range(100101):
	i = randint(0,60000)
	nastaviZacetniNivo(i)
	zeljeniNivoji = zeljeniNivojiNastavi(i)
	popraviVse()
	popraviDelte(zeljeniNivoji)
	enoUcenje()
	if j%1000 == 0:
		print(str(int((j/1000)))+"%")

	if j % 2000 == 0:
		shraniVse()
	if j > 100000:
		prou = False
		prou = (zeljeniNivoji.index(max(zeljeniNivoji))==np.argmax(vsiNivoji[-1].vektor_vrednosti))
		if prou:
			pravilne += 1


print("PRAVILNE: "+str(pravilne)+"/100")
print("="*60)

print(zeljeniNivoji.index(max(zeljeniNivoji)))

print("NA KONCU je zadnji:")
print(vsiNivoji[-1].vektor_vrednosti)
print(np.argmax(vsiNivoji[-1].vektor_vrednosti))
print("*"*70)
print(vsiNivoji[1].delte)

shraniVse()
