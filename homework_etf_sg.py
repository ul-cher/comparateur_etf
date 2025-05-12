import random, string
import numpy as np
from dataclasses import dataclass
from typing import List, Dict
import streamlit as st
import pandas as pd

MIN_PRICE = 60
MAX_PRICE = 120
MIN_CASH = -1
MAX_CASH = 1


@dataclass
class Bond:
    id: str
    price: float
    weight: float

    def __post_init__(self):
        if not (MIN_PRICE < self.price < MAX_PRICE):
            raise ValueError(f"Le prix d'obligation doit etre entre {MIN_PRICE} et {MAX_PRICE}")
        if not (0 < self.weight < 1):
            raise ValueError("Le poids d'obligation doit etre entre (0, 1)")

@dataclass 
class ETFComposition:
    date: str
    bonds: List[Bond]
    cash: float

    def __post_init__(self):
        if not (MIN_CASH < self.cash < MAX_CASH):
            raise ValueError(f"Le cash doit etre entre {MIN_CASH} et {MAX_CASH}")
        
        total_weight = sum(b.weight for b in self.bonds)
        if not np.isclose(total_weight, 1.0, atol=0.001):
            raise ValueError(f"La somme des poids est {round(total_weight, 3)}. Elle doit etre égale à 1.0.")

    #calculer le prix de l'ETF par la formule 
    def calculate_price(self):
        return round(sum(b.price * b.weight for b in self.bonds) + self.cash,3)

    #calculer le prix de l'ETF par la produit scalaire entre deux vecteurs prices et weights
    # def calculate_price(self):
    #     prices = np.array([b.price for b in self.bonds])
    #     weights = np.array([b.weight for b in self.bonds])
    #     return round(np.dot(prices, weights) + self.cash, 3)


#Générer un bond
def generate_bond(bond_weigth: float) -> Bond:
    return Bond(
        id= ''.join(random.choices(string.ascii_uppercase, k=5)), 
        price = round(random.uniform(60,120), 3),
        weight= bond_weigth
    )

#Générer des bonds en respectant la regle pour les poids (sum(poids)=1)
def generate_new_bonds(num_bonds: int) -> List[Bond]:
    if num_bonds <= 0:
        raise ValueError("Le nombre d'obligations doit etre supérieur à 0")
        
    weights = np.zeros(num_bonds)
    remaining_weight = 1.0
    for i in range(num_bonds-1):
        w = round(random.uniform(0.01, remaining_weight - 0.01*(num_bonds-i-1)), 3)
        weights[i] = w
        remaining_weight -= w
    weights[-1] = round(remaining_weight, 3)
        
    if abs(sum(weights)-1.0) > 0.001:
        raise ValueError("La somme des poids n'est pas égale à 1 après la génération")
            
    bonds = []
    for w in weights:
        try:
            bonds.append(generate_bond(w))
        except ValueError as e:
            st.error(f"Erreur lors de la génération d'une obligation: {str(e)}")
            raise
                
    return bonds

#Générer une composistion d'ETF
def generate_etf_composition(date: str, num_bonds: int) -> ETFComposition:
    bonds = generate_new_bonds(num_bonds)
    cash = round(random.uniform(-1,1), 3)
    return ETFComposition(date=date, bonds=bonds, cash=cash)

#Verification de la composition
def validate_composition(comp: ETFComposition) -> bool:
    try:
        if any(not (MIN_PRICE < b.price < MAX_PRICE) for b in comp.bonds):
            raise ValueError(f"Le prix d'obligation doit etre entre {MIN_PRICE} et {MAX_PRICE}")
        
        if any(not (0< b.weight < 1) for b in comp.bonds):
            raise ValueError("Le poids d'obligation doit etre entre (0, 1)")
            
        if not np.isclose(sum(b.weight for b in comp.bonds), 1.0, atol=0.001):
            raise ValueError("La somme des poids doit etre égale à 1.0")
            
        if not (MIN_CASH < comp.cash < MAX_CASH):
            raise ValueError(f"Le cash doit etre entre {MIN_CASH} et {MAX_CASH}")
            
        return True
    except Exception as e:
        st.error(f"La composition de l'ETF ne respect pas des contraintes: {str(e)}")
        return False

#Comparer deux composistions
def compare_compositions(comp1: ETFComposition, comp2: ETFComposition) -> Dict:
    bonds1 = {b.id: b for b in comp1.bonds}
    bonds2 = {b.id: b for b in comp2.bonds}

    common_ids = bonds1.keys()&bonds2.keys()
    incoming_ids = bonds2.keys()-bonds1.keys()
    outgoing_ids = bonds1.keys()-bonds2.keys()

    common_bonds = []
    for id in common_ids:
        weight_diff = bonds2[id].weight-bonds1[id].weight
        common_bonds.append({
            'id': id,
            'weight_variation': round(weight_diff, 3),
            'old_weight': round(bonds1[id].weight,3),
            'new_weight': round(bonds2[id].weight,3)
        })

    incoming = [{'id': id,'price': bonds2[id].price, 'weight': bonds2[id].weight} for id in incoming_ids]
    outgoing = [{'id': id,'price': bonds1[id].price, 'weight': bonds1[id].weight} for id in outgoing_ids]
    return {
        'common': common_bonds,
        'incoming': incoming,
        'outgoing': outgoing,
        'cash_variation': round(comp2.cash-comp1.cash, 3)
    }


#-------- STREAMLIT --------
st.set_page_config(page_title="ETF Comparateur", layout="wide")
st.title("Comparateur de deux compositions ETF")

#Compositions initiales sauvegardées dans la session state
if 'comp_8' not in st.session_state:
    st.session_state.comp_8 = generate_etf_composition("8 mai 2025", 3)
if 'comp_9' not in st.session_state:
    st.session_state.comp_9 = generate_etf_composition("9 mai 2025", 3)

#Editable form pour une composistion de l'ETF
def edit_etf(label: str, etf: ETFComposition) -> ETFComposition:
    st.subheader(f"Composition du {etf.date}")
    num_bonds = st.slider(f"Nombre d'obligations ({label})", 3, 10, value=len(etf.bonds), key=f"{label}_num_bonds")
    #si le nombre d'obligations a changé, de nouvelles obligations sont générées
    if len(etf.bonds) != num_bonds:
        etf.bonds = generate_new_bonds(num_bonds)
   
    #affichage des compositions
    bond_data = pd.DataFrame([{"id": b.id, "price": b.price, "weight": b.weight} for b in etf.bonds])
    edited_df = st.data_editor(bond_data, num_rows="fixed", key=f"{label}_editor")

    total_weight = sum(edited_df["weight"])
    if not np.isclose(total_weight, 1.0, atol=0.001):
        st.error(f"La somme des poids est {round(total_weight, 3)}. Elle doit etre égale à 1.0.")
        st.stop()

    normalized_bonds = []
    for i, row in edited_df.iterrows():
        normalized_bonds.append(Bond(
            id=row["id"],
            price=round(float(row["price"]),3),
            weight=round(float(row["weight"]), 3)
        ))

    cash = st.number_input(f"Cash ({label})",value=float(etf.cash),min_value=float(MIN_CASH+0.01),max_value=float(MAX_CASH-0.01),step=0.01, key=f"{label}_cash")
    new_etf = ETFComposition(date=etf.date, bonds=normalized_bonds, cash=round(cash,3))
    if validate_composition(new_etf):
        return new_etf  
    return etf


#Columns pour les deux compositions
col1, col2 = st.columns(2)
with col1:
    st.session_state.comp_8 = edit_etf("8 mai", st.session_state.comp_8)
with col2:
    st.session_state.comp_9 = edit_etf("9 mai", st.session_state.comp_9)

#Display des prix calcules
price_8 = st.session_state.comp_8.calculate_price()
price_9 = st.session_state.comp_9.calculate_price()
st.subheader("Prix de chaque composition")
st.write(f"**Prix du 8 mai 2025**: {price_8} EUR | Cash: {st.session_state.comp_8.cash} EUR")
st.write(f"**Prix du 9 mai 2025**: {price_9} EUR | Cash: {st.session_state.comp_9.cash} EUR")


#--Les résultats de la comparaison--
st.markdown("---")
st.subheader("Comparaison des compositions")
comparison = compare_compositions(st.session_state.comp_8, st.session_state.comp_9)

#les variations de poids pour les bonds communs
if comparison['common']:
    for bond in comparison['common']:
        st.write(f"ID: {bond['id']} | Variation du poids: {bond['weight_variation']} | Ancien: {bond['old_weight']} -> Nouveau: {bond['new_weight']}")
    st.subheader("Tableau des obligations communes")
    st.table(pd.DataFrame(comparison['common']))
else:
    st.write("Pas d'bligations communes.")

#les bonds entrants/sortants
if comparison['incoming']:
    st.write(f"Obligations entrantes: {[b['id'] for b in comparison['incoming']]}")
    st.subheader("Tableau des obligations entrantes")
    st.table(pd.DataFrame(comparison['incoming']))
else:
    st.write("Pas d'bligations entrantes.")

if comparison['outgoing']:
    st.write(f"Obligations sortantes: {[b['id'] for b in comparison['outgoing']]}")
    st.subheader("Tableau des obligations sortantes")
    st.table(pd.DataFrame(comparison['outgoing']))
else:
    st.write("Pas d'bligations sortantes.")

#la variation du cash
st.write(f"Variation du cash: {comparison['cash_variation']}EUR")