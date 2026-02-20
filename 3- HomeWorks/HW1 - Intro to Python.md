# Homework 1 — Intro to Python
### Case Study: **QuickMart — A Local Supermarket Sales Tracker**

---

## Background

You have been hired as a junior data analyst at **QuickMart**, a small local supermarket. The store manager wants to digitize the store's operations — from tagging products to analyzing weekly sales. Your job is to build this system step by step using everything you learned in the Intro to Python sessions.

Each part of this homework builds on the previous one, so work through them in order.

---

## Part 1 — Variables & Data Types

The store manager gave you information about one product to start with.

### Task 1.1 — Define the Product

Create the following variables for a product called **"Olive Oil"**:

| Variable | Value |
|---|---|
| `product_name` | `"Olive Oil"` |
| `price` | `8.75` |
| `quantity_in_stock` | `120` |
| `is_organic` | `True` |
| `product_code` | `"OO-2024-EG"` |

Then print each variable along with a descriptive label.  
**Expected output (example):**
```
Product Name: Olive Oil
Price: 8.75
Quantity in Stock: 120
Organic: True
Product Code: OO-2024-EG
```

<details>
<summary>💡 Hint</summary>

Use `print()` with string concatenation or commas to label your output.  
Remember: mixing strings and numbers requires converting numbers to strings using `str()`, or you can separate them with a comma inside `print()`.

```python
print("Price:", price)
```

</details>

---

### Task 1.2 — Type Check & Conversion

1. Print the **data type** of each variable you created above using `type()`.
2. The manager wants to store the price as an **integer** (he rounds down for internal use). Convert `price` to an integer and store it in a variable called `price_int`. Print both the original and the converted value.
3. A new field `is_available` should be derived: convert `quantity_in_stock` to a **boolean**. What do you get if the stock is `0`? Test both cases and explain the result with a comment.

<details>
<summary>💡 Hint</summary>

- `type(variable)` returns the data type.
- `int(8.75)` truncates (floors) the decimal — it does **not** round.
- In Python, any non-zero number converted to `bool` is `True`; `0` becomes `False`.

```python
price_int = int(price)
bool(0)    # → False
bool(120)  # → True
```

</details>

---

## Part 2 — Conditions & Branching

The manager wants the system to make smart decisions about pricing and stock.

### Task 2.1 — Budget Check

A customer has a budget of **`50.00`** EGP. Write a condition that checks:
- If the price of Olive Oil is **within** the customer's budget → print `"Affordable — added to cart!"`
- Otherwise → print `"Over budget."`

<details>
<summary>💡 Hint</summary>

Use a simple `if / else` block comparing `price` to the budget variable.

```python
budget = 50.00
if price <= budget:
    ...
```

</details>

---

### Task 2.2 — Discount Logic

QuickMart applies discounts based on the quantity a customer orders. Write an `if / elif / else` block that prints the right discount message given a variable `order_qty`:

| Condition | Discount |
|---|---|
| `order_qty >= 50` | 20% discount |
| `order_qty >= 20` | 10% discount |
| `order_qty >= 10` | 5% discount |
| Less than 10 | No discount |

Also **calculate and print the final price** after applying the discount.

<details>
<summary>💡 Hint</summary>

Apply the discount as a multiplier:

```python
final_price = price * (1 - 0.10)   # 10% off
```

Chain your conditions from the **largest** to the **smallest** threshold using `elif`.

</details>

---

### Task 2.3 — Combined Stock & Price Check

Using **logical operators**, write a single condition that checks:
> The product is in stock (`quantity_in_stock > 0`) **AND** the price is less than `100`

- If both are true → `"Ready to sell!"`
- If only one is true → `"Partially available — check conditions."`
- If neither is true → `"Cannot sell this product."`

> *Tip: You'll need to think about how to handle the "only one is true" case with `elif`.*

<details>
<summary>💡 Hint</summary>

Use `and`, `or`, and `not` operators.

```python
if quantity_in_stock > 0 and price < 100:
    ...
elif quantity_in_stock > 0 or price < 100:
    ...
else:
    ...
```

</details>

---

## Part 3 — Loops

The store has more than one product. Time to iterate!

### Task 3.1 — Price Tags with a `for` Loop

You have the following list of prices (use it as-is):

```python
prices = [8.75, 15.00, 3.50, 22.00, 6.99, 45.00, 1.25]
```

Using a `for` loop:
1. Print each price.
2. Apply a **flat 15% discount** to every price and print the discounted value next to the original.

<details>
<summary>💡 Hint</summary>

```python
for p in prices:
    discounted = p * 0.85
    print("Original:", p, "→ Discounted:", round(discounted, 2))
```

Use `round(value, 2)` to keep two decimal places.

</details>

---

### Task 3.2 — Stock Countdown with a `while` Loop

A customer wants to keep buying units of a product priced at **`8.75`** each. Their starting budget is **`50.00`**.

Write a `while` loop that:
1. Keeps buying one unit at a time as long as the customer can afford it.
2. Prints the number of units bought and the remaining budget after each purchase.
3. After the loop ends, prints the **total units purchased** and the **leftover budget**.

<details>
<summary>💡 Hint</summary>

Deduct the price from the budget inside the loop and count iterations with a counter variable.

```python
budget = 50.00
units_bought = 0
while budget >= price:
    budget -= price
    units_bought += 1
```

</details>

---

### Task 3.3 — Loop + Condition: Find Expensive Items

Using the same `prices` list from Task 3.1, loop through and print only the prices that are **above 10.00**. Count how many such items there are and print the count at the end.

<details>
<summary>💡 Hint</summary>

Use an `if` statement inside your `for` loop and a counter variable that starts at `0`.

```python
count = 0
for p in prices:
    if p > 10:
        count += 1
        print(p)
```

</details>

---

## Part 4 — Strings

The manager now wants the system to generate readable product labels and receipts.

### Task 4.1 — Product Label Formatting

Using the `product_code` variable `"OO-2024-EG"` from Task 1.1:

1. Print the **first two characters** (the category code).
2. Print the **last two characters** (the country code).
3. Print the **year** embedded in the code (characters at index 3 to 6).
4. Print the whole code **in reverse** using stride.

<details>
<summary>💡 Hint</summary>

Python string indexing: `s[start:end:step]`

```python
product_code[0:2]      # first two characters
product_code[-2:]      # last two characters
product_code[3:7]      # year portion
product_code[::-1]     # reverse
```

</details>

---

### Task 4.2 — String Methods

Using `product_name = "Olive Oil"`:

1. Convert it to **uppercase** and print.
2. Convert it to **lowercase** and print.
3. Check if it **starts with** `"Olive"` and print the result.
4. **Replace** `"Oil"` with `"Spread"` and print the new name.
5. **Split** the product name into a list of words and print the list.

<details>
<summary>💡 Hint</summary>

String methods: `.upper()`, `.lower()`, `.startswith()`, `.replace()`, `.split()`

```python
product_name.replace("Oil", "Spread")
product_name.split()   # splits on spaces by default
```

</details>

---

### Task 4.3 — Receipt Generator (Strings + Loops + Conditions)

You have two lists:

```python
items     = ["Olive Oil", "Milk", "Bread", "Eggs", "Cheese"]
item_prices = [8.75, 5.50, 3.00, 12.00, 18.50]
```

Write a loop that builds and prints a receipt in this format:

```
===== QuickMart Receipt =====
1. Olive Oil          8.75 EGP
2. Milk               5.50 EGP
3. Bread              3.00 EGP
4. Eggs              12.00 EGP
5. Cheese            18.50 EGP
-----------------------------
TOTAL:               47.75 EGP
=============================
```

- Items costing more than **15 EGP** should have `⭐` appended to their name.
- Calculate and print the total at the end.

<details>
<summary>💡 Hint</summary>

- Use `enumerate()` to get the index and item together.
- Use f-strings or string formatting to align columns: `f"{name:<20}{price:>6.2f}"`.
- Accumulate the total with a running sum variable.

```python
total = 0
for i, (item, p) in enumerate(zip(items, item_prices), start=1):
    total += p
    label = item + " ⭐" if p > 15 else item
    print(f"{i}. {label:<22}{p:>6.2f} EGP")
print(f"TOTAL: {total:.2f} EGP")
```

</details>

---

## Part 5 — Lists

QuickMart now tracks a full shopping cart.

### Task 5.1 — Build & Modify a Shopping Cart

Start with this shopping cart:

```python
cart = ["Olive Oil", "Milk", "Bread"]
```

Perform the following operations **in order**, printing the cart after each step:

1. **Append** `"Eggs"` to the cart.
2. **Insert** `"Butter"` at index `1`.
3. **Remove** `"Milk"` from the cart.
4. Print the **length** of the cart.
5. Print the item at **index 2**.
6. Print the cart in **reverse order** (do not modify the original; use slicing).

<details>
<summary>💡 Hint</summary>

- `.append(item)` adds to the end.
- `.insert(index, item)` inserts at a specific position.
- `.remove(item)` removes the first match.
- `len(cart)` gives the size.
- `cart[::-1]` creates a reversed copy.

</details>

---

### Task 5.2 — Price Analysis on a List

Given:

```python
item_prices = [8.75, 5.50, 3.00, 12.00, 18.50, 7.25, 9.99]
```

Without using built-in functions like `max()` or `min()`:
1. Find and print the **most expensive** item price using a loop.
2. Find and print the **cheapest** item price using a loop.
3. Calculate and print the **average** price.

<details>
<summary>💡 Hint</summary>

Initialize `highest = item_prices[0]` and `lowest = item_prices[0]`, then loop and compare.

```python
highest = item_prices[0]
for p in item_prices:
    if p > highest:
        highest = p
```

</details>

---

### Task 5.3 — Filtering a List (Lists + Loops + Conditions)

Using the `items` and `item_prices` lists from Task 4.3, build a **new list** called `affordable` that contains only the item names where the price is **less than or equal to 10 EGP**.

Print the `affordable` list and how many items are in it.

<details>
<summary>💡 Hint</summary>

Create an empty list and append to it inside the loop:

```python
affordable = []
for i, p in enumerate(item_prices):
    if p <= 10:
        affordable.append(items[i])
```

</details>

---

## Part 6 — Dictionaries

The store needs a proper product catalog.

### Task 6.1 — Build the Product Catalog

Create a dictionary called `catalog` where each **key** is a product name and each **value** is its price:

| Product | Price |
|---|---|
| Olive Oil | 8.75 |
| Milk | 5.50 |
| Bread | 3.00 |
| Eggs | 12.00 |
| Cheese | 18.50 |

Then:
1. Print the price of `"Eggs"`.
2. Add a new product: `"Butter"` → `9.00`.
3. Update the price of `"Cheese"` to `20.00`.
4. Delete `"Bread"` from the catalog.
5. Print all the **keys** (product names).
6. Print all the **values** (prices).

<details>
<summary>💡 Hint</summary>

```python
catalog["Eggs"]            # access
catalog["Butter"] = 9.00  # add
catalog["Cheese"] = 20.00 # update
del catalog["Bread"]       # delete
catalog.keys()
catalog.values()
```

</details>

---

### Task 6.2 — Loop Over the Catalog

Loop over the `catalog` dictionary and:
1. Print each product and its price in a readable format: `"Olive Oil → 8.75 EGP"`.
2. Count how many products cost **more than 10 EGP** and print that count.

<details>
<summary>💡 Hint</summary>

Use `.items()` to iterate over key-value pairs:

```python
for product, price in catalog.items():
    print(f"{product} → {price} EGP")
```

</details>

---

### Task 6.3 — Nested Dictionary (Dicts + Lists + Conditions)

The manager wants richer product data. Upgrade the `catalog` so each product maps to a **dictionary** with `"price"`, `"stock"`, and `"category"` keys. Use the data below:

```python
catalog = {
    "Olive Oil": {"price": 8.75,  "stock": 120, "category": "Pantry"},
    "Milk":      {"price": 5.50,  "stock": 0,   "category": "Dairy"},
    "Eggs":      {"price": 12.00, "stock": 60,  "category": "Dairy"},
    "Cheese":    {"price": 20.00, "stock": 15,  "category": "Dairy"},
    "Bread":     {"price": 3.00,  "stock": 200, "category": "Bakery"},
}
```

Write code that loops through the catalog and prints a **status report** for each product:

```
Olive Oil  | Price: 8.75  | Stock: 120 | Status: Available
Milk       | Price: 5.50  | Stock: 0   | Status: OUT OF STOCK
...
```

- Status is `"Available"` if `stock > 0`, otherwise `"OUT OF STOCK"`.

<details>
<summary>💡 Hint</summary>

Access nested values with chained keys:

```python
for name, info in catalog.items():
    status = "Available" if info["stock"] > 0 else "OUT OF STOCK"
    print(f"{name} | Price: {info['price']} | Stock: {info['stock']} | Status: {status}")
```

</details>

---

## Part 7 — Pandas: Analyzing Real Sales Data

QuickMart has a full CSV file of Chicago sales census data. Time to put your analytical skills to work using **pandas**.

> **Dataset files (located in the `Data/` folder):**
> - `ChicagoCensusData.csv`
> - `ChicagoCrimeData.csv`

---

### Task 7.1 — Load & Inspect

1. Import pandas as `pd`.
2. Load `ChicagoCensusData.csv` into a DataFrame called `df_census`.
3. Load `ChicagoCrimeData.csv` into a DataFrame called `df_crime`.
4. For **each** DataFrame, print:
   - Its **shape** (rows × columns).
   - The **first 5 rows** using `.head()`.
   - The **column names**.
   - Basic statistics using `.describe()`.

<details>
<summary>💡 Hint</summary>

```python
import pandas as pd

df_census = pd.read_csv("../Data/ChicagoCensusData.csv")
df_crime  = pd.read_csv("../Data/ChicagoCrimeData.csv")

print(df_census.shape)
df_census.head()
df_census.columns
df_census.describe()
```

Adjust the path based on where your notebook lives relative to the `Data/` folder.

</details>

---

### Task 7.2 — Filter & Sort

Using `df_census`:

1. Filter rows where `PER CAPITA INCOME` is **greater than 30,000** and store the result in `high_income`.
2. Print the number of such communities.
3. Sort `high_income` by `PER CAPITA INCOME` in **descending** order and print the top 5.

<details>
<summary>💡 Hint</summary>

```python
high_income = df_census[df_census["PER CAPITA INCOME"] > 30000]
high_income.sort_values("PER CAPITA INCOME", ascending=False).head(5)
```

Column names may vary — use `df_census.columns` to check exact names.

</details>

---

### Task 7.3 — Group & Aggregate

Using `df_crime`:

1. Find the **top 5 most common crime types** (by count). *(Hint: look for a column like `PRIMARY TYPE`.)*
2. Print the results as a table with crime type and count.
3. Find the **location description** where the most crimes happened.

<details>
<summary>💡 Hint</summary>

```python
df_crime["PRIMARY TYPE"].value_counts().head(5)
df_crime["LOCATION DESCRIPTION"].value_counts().idxmax()
```

</details>

---

### Task 7.4 — Putting It All Together (Full Analysis)

Answer the following questions using pandas. **Print a clear answer for each one.**

1. What is the **average per capita income** across all community areas?
2. How many crime records have `ARREST` equal to `True`?
3. Which community in the census data has the **highest** `PERCENT HOUSEHOLDS BELOW POVERTY`?
4. **Merge** the two DataFrames on the `COMMUNITY AREA NUMBER` column (make sure column names match — rename if needed). How many rows does the merged DataFrame have?
5. From the merged DataFrame, find the **average per capita income** for communities where a crime `ARREST` was made vs. where it was not. What do you observe?

<details>
<summary>💡 Hint</summary>

```python
# Average income
df_census["PER CAPITA INCOME"].mean()

# Arrest count
df_crime["ARREST"].sum()   # True is treated as 1

# Highest poverty
df_census.loc[df_census["PERCENT HOUSEHOLDS BELOW POVERTY"].idxmax(), "COMMUNITY AREA NAME"]

# Merge
merged = pd.merge(df_crime, df_census, left_on="COMMUNITY AREA NUMBER", right_on="COMMUNITY AREA NUMBER")

# Average income by arrest status
merged.groupby("ARREST")["PER CAPITA INCOME"].mean()
```

Check exact column names in each DataFrame before merging — they must be identical or mapped explicitly.

</details>

---

## Submission Checklist

Before submitting, make sure your notebook:

- [ ] Has **all 7 parts** completed with working code cells.
- [ ] Prints **clear, labelled output** for each task.
- [ ] Includes **comments** explaining your reasoning where appropriate.
- [ ] Has no crash-causing errors when run top to bottom with **Restart & Run All**.

---

> **Good luck! Remember: every expert was once a beginner.** 🚀
