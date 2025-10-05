# **DOCUMENT D’ENTRÉE EN RELATION (DER)**  
*(Conformément aux articles L.520-1 et suivants du Code des assurances et L.541-8-1 du Code monétaire et financier)*  

---

## **1. Identification du courtier**

- **Dénomination sociale :** {{ DER_courtier.nom_cabinet }}  
- **Responsable :** {{ DER_courtier.nom_responsable }}  
- **Statut social :** {{ DER_statut_social.lib }}  
- **Capital social :** {{ DER_courtier.capital_social }} €  
- **SIREN :** {{ DER_courtier.siren }}  
- **RCS :** {{ DER_courtier.rcs }}  
- **Numéro ORIAS :** {{ DER_courtier.numero_orias }}  
- **Adresse :** {{ DER_courtier.adresse_rue }}, {{ DER_courtier.adresse_cp }} {{ DER_courtier.adresse_ville }}  
- **Courriel :** {{ DER_courtier.courriel }}  
- **Association professionnelle :** {{ DER_courtier.association_prof }} (adhérent n° {{ DER_courtier.num_adh_assoc }})  
- **Catégorie de courtage :** {{ DER_courtier.categorie_courtage }}  

---

## **2. Activités et domaines d’exercice**

Le cabinet exerce les activités suivantes :  

{% for activite in DER_courtier_activite %}
- **{{ activite.libelle }}** ({{ activite.domaine }}) – Statut : **{{ activite.statut }}**
{% endfor %}

Les activités non exercées sont mentionnées à titre informatif.  

---

## **3. Autorités de tutelle**

- **AMF (Autorité des Marchés Financiers)**  
  17, place de la Bourse – 75082 Paris Cedex 02  
- **ACPR (Autorité de Contrôle Prudentiel et de Résolution)**  
  61, rue Taitbout – 75436 Paris Cedex 09  

Ces autorités veillent au respect des obligations professionnelles du courtier.  

---

## **4. Nature et étendue du service**

{{ DER_courtier.nom_cabinet }} agit en qualité de **courtier indépendant**.  
Son rôle est d’analyser **vos besoins, vos objectifs et votre profil de risque** afin de vous recommander les solutions les plus adaptées à votre situation.  

Le cabinet ne perçoit ni ne conserve de fonds destinés aux assureurs ou aux clients.  

---

## **5. Rémunération et frais**

Le cabinet est rémunéré sous forme :
- de **commissions** versées par les compagnies partenaires ;  
- et/ou de **frais** ou **honoraires** prévus dans la lettre de mission.

### Modes de facturation :
{% for f in DER_courtier_mode_facturation %}
- **{{ f.type }} :** {% if f.montant %}{{ f.montant }} €{% elif f.pourcentage %}{{ f.pourcentage }} %{% endif %}
{% endfor %}

### Modes de communication :
{% for c in DER_courtier_ref_mode_comm %}
- {{ c.lib }}
{% endfor %}

---

## **6. Garanties professionnelles**

Le cabinet justifie d’une **assurance responsabilité civile professionnelle** et d’une **garantie financière**, conformément à la réglementation.  

| Type de garantie                             | Montant Assurance (IAS) | Montant Banque (IOBSP) |
| -------------------------------------------- | ----------------------- | ---------------------- |
| {% for g in DER_courtier_garanties_normes %} |                         |                        |
| {{ g.type_garantie }}                        | {{ g.IAS }} €           | {{ g.IOBSP }} €        |
| {% endfor %}                                 |                         |                        |

---

## **7. Données personnelles**

Vos informations sont collectées et traitées selon le **Règlement Général sur la Protection des Données (RGPD)** et la **loi Informatique et Libertés**.  
Elles sont utilisées pour analyser votre situation, préparer des recommandations et assurer le suivi de vos contrats.  

Vous disposez d’un droit d’accès, de rectification et de suppression de vos données.  
Pour exercer ces droits, contactez : **{{ DER_courtier.responsable_dpo }} – {{ DER_courtier.courriel }}**

---

## **8. Médiation et réclamations**

En cas de réclamation, vous pouvez contacter :  
**{{ DER_courtier.nom_cabinet }} – Service Réclamations**  
{{ DER_courtier.adresse_rue }}, {{ DER_courtier.adresse_cp }} {{ DER_courtier.adresse_ville }}  
Email : {{ DER_courtier.courriel }}

En cas de désaccord non résolu, vous pouvez saisir :  
- **{{ DER_courtier.centre_mediation }}**  
  {{ DER_courtier.mediators }}  
  Contact : {{ DER_courtier.mail_mediators }}

---

## **9. Identification du client**

- **Qualité :** {{ mariadb_clients.qualite }}  
- **Nom :** {{ mariadb_clients.nom }}  
- **Prénom :** {{ mariadb_clients.prenom }}  
- **Date de naissance :** {{ mariadb_clients.date_naissance }}  
- **Lieu de naissance :** {{ mariadb_clients.lieu_naissance_ville }} ({{ mariadb_clients.lieu_naissance_cp }} – {{ mariadb_clients.lieu_naissance_pays }})  
- **Nationalité :** {{ mariadb_clients.nationalite }}  
- **Numéro fiscal (NIF) :** {{ mariadb_clients.nif }}  
- **Adresse principale :** {{ mariadb_clients.adresse_rue }}, {{ mariadb_clients.adresse_cp }} {{ mariadb_clients.adresse_ville }}, {{ mariadb_clients.adresse_pays }}  
- **Adresse postale :** {{ mariadb_clients.adresse_postale }}  
- **Téléphone :** {{ mariadb_clients.telephone }}  
- **Email :** {{ mariadb_clients.email }}  
- **Situation maritale :** {{ mariadb_clients.situation_maritale }}  
- **Profession :** {{ mariadb_clients.profession }}  
- **Secteur d’activité :** {{ mariadb_clients.secteur_id }}  
- **Niveau SRRI :** {{ mariadb_clients.srri }}

---

## **10. Acceptation**

Je, soussigné(e) **{{ mariadb_clients.prenom }} {{ mariadb_clients.nom }}**, reconnais avoir reçu et pris connaissance du présent **Document d’Entrée en Relation** avant tout engagement.  

**Fait à :** {{ DER_courtier.adresse_ville }}  
**Le :** {{ date_signature }}  

**Signature du client :** ________________________  
**Signature du courtier :** ________________________