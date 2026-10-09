Time and identifiers

Column	Role	Meaning	Notes
order date (DateOrders)	Time index	Order timestamp	Text, parse to datetime. Keep 2015-01 to 2017-09
Product Card Id	Entity id	Product (118 total)	Use as the series id
Product Name	Label	Product name	Display only
Category Id	Feature	Product category (50 names, 51 ids)	Electronics has two ids
Department Name	Feature	Department (11)	Context

Demand and price

Column	Role	Meaning	Notes
Order Item Quantity	Demand target	Units on the line (1 to 5)	Summed to weekly units per product
Order Item Product Price	Feature	List price of the product	Constant per product, so no price change
Order Item Discount Rate	Candidate feature	Discount fraction (18 values)	Evenly spread, so test it in an ablation

Shipping and context

Column	Role	Meaning	Notes
Shipping Mode	Feature, decision variable	First Class, Same Day, Second Class, Standard Class	Candidate shipping decision
Days for shipment (scheduled)	Feature	Promised days (known at order time)	Usable
Days for shipping (real)	Outcome only	Actual days	Not an input for demand. Used to simulate lead-time risk
Market, Order Region, Customer Segment, Type	Feature	Geography, segment, payment type	Context for cost model

Cost

Column	Role	Meaning	Notes
implied_unit_cost (derived)	Cost target	(Order Item Total − Order Profit Per Order) / Quantity	Derived, not observed cost

Excluded from cost-model inputs (they build the target or reveal it)

Column	Reason
Sales, Order Item Total, Order Item Discount	Part of the target formula
Order Profit Per Order, Benefit per order, Order Item Profit Ratio	Part of the target formula
Sales per customer	Same values as Order Item Total

Excluded from demand inputs (known only after the event)

Column	Reason
Delivery Status, Late_delivery_risk, Days for shipping (real), shipping date, Order Status	Happen after the order

Dropped as unusable

Column	Reason
Product Description	100% missing
Order Zipcode	86% missing
Customer Email, Customer Password	Masked, one value
Product Status	Constant
Customer Fname, Lname, Street	Personal, no modelling value
Duplicates (Order Customer Id, Order Item Cardprod Id, Product Category Id, Product Price)	Same information as another column

Simulated (not in the data, always labelled)

Quantity	Treatment
Stock level, supplier, supplier capacity	Assumption
Ordering cost, holding cost rate, stockout penalty	Assumption, with a sensitivity range