Comparateur de Compositions ETF
Cette application permet de générer et comparer deux compositions d’un ETF à deux dates différentes (8 et 9 mai)

Fonctionnalités:
Génération aléatoire d'ETF-obligataires respectant les contraintes de poids et de prix.
Calcul automatique du prix total de l’ETF.

Comparaison détaillée:
Obligations communes avec variation des poids.
Obligations entrantes/sortantes.
Variation du cash entre les deux dates.

Contraintes respectées
Prix d'une obligation: entre 60 et 120
Poids individuel: entre 0 et 1
Somme des poids = 1.0
Cash: entre -1 et 1

generate_bond(bond_weigth: float) -> Bond
 - Générer un bond avec un code unique et un prix aléatoire entre 60 et 120
generate_new_bonds(num_bonds: int) -> List[Bond]
 - Générer des bonds respectant les contraintes de poids
generate_etf_composition(date: str, num_bonds: int) -> ETFComposition
 - Génère une composition d'ETF 
validate_composition(comp: ETFComposition) -> bool
 - Vérifie si la composition est valide: poids entre 0 et 1, somme des poids = 1.0, cash entre -1 et 1.
calculate_price(self)
 - Calcule le prix total de l’ETF en sommant les produits prix * poids + cash.
compare_compositions(comp1: ETFComposition, comp2: ETFComposition) -> Dict
 - Compare deux compositions et retourne les obligations communes (avec variation), entrantes, sortantes, et la variation du cash.

Lancer l'application: 
pip install streamlit numpy pandas
streamlit run homework_etf_sg.py

homework_etf_sg.py : script principal contenant toute la logique de génération, calcul et comparaison.