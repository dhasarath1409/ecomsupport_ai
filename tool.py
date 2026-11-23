import json
from langchain_core.tools import tool

with open("data/orders.json") as f:
    ORDERS_DB = json.load(f)


@tool
def lookup_order(order_id: str) -> str:
    """
    Look up the status of a customer's order by order ID.
    Use this whenever the customer asks about order status,
    delivery, tracking, or shipment for a specific order number.
    """
    order = ORDERS_DB.get(order_id)
    if not order:
        return f"No order found with ID {order_id}. Please double-check the order number."

    parts = [f"Order {order_id}: {order['item']} -- Status: {order['status']}"]
    if order["eta"]:
        parts.append(f"Expected delivery: {order['eta']}")
    if order["tracking"]:
        parts.append(f"Tracking ({order['carrier']}): {order['tracking']}")
    return " | ".join(parts)

if __name__ == "__main__":
    print(lookup_order.invoke({"order_id": "1002"}))  # test