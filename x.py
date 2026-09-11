import requests
from playwright.sync_api import sync_playwright

CLIENT_ID = ""
CLIENT_SECRET = ""
# ============================================================
# 1. Obter token
# ============================================================

token_response = requests.post(
    "https://oauth.livepix.gg/oauth2/token",
    data={
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "scope": "payments:write"
    }
)

print("Status token:", token_response.status_code)
print("Resposta token:", token_response.text)

token_response.raise_for_status()

access_token = token_response.json()["access_token"]


# ============================================================
# 2. Criar pagamento
# ============================================================

payment_response = requests.post(
    "https://api.livepix.gg/v2/payments",
    headers={
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    },
    json={
        "amount": 1000,  # R$ 10,00
        "currency": "BRL",
        "redirectUrl": "https://seusite.com/pagamento/retorno"
    }
)

print("\nStatus pagamento:", payment_response.status_code)
print("Resposta pagamento:", payment_response.text)

payment_response.raise_for_status()

payment = payment_response.json()

print("\nDados recebidos:", payment)


# ============================================================
# 3. Pegar dados do pagamento
# ============================================================

payment_data = payment["data"]

reference = payment_data["reference"]
redirect_url = payment_data["redirectUrl"]

print("\nReference:", reference)
print("Redirect URL:", redirect_url)


# ============================================================
# 4. Abrir o checkout com Playwright
# ============================================================

with sync_playwright() as p:

    browser = p.chromium.launch(
        headless=False
    )

    page = browser.new_page(
        viewport={
            "width": 1280,
            "height": 900
        }
    )

    print("\nAbrindo checkout...")

    page.goto(
        redirect_url,
        wait_until="networkidle",
        timeout=60000
    )

    print("Checkout carregado.")

    # ========================================================
    # 5. Clicar em "Continuar"
    # ========================================================

    continuar = page.get_by_role(
        "button",
        name="Continuar"
    )

    continuar.wait_for(
        state="visible",
        timeout=30000
    )

    print("Botão Continuar encontrado.")

    continuar.click()

    print("Continuar clicado.")


    # ========================================================
    # 6. Esperar campo do PIX
    # ========================================================

    pix_input = page.locator(
        'input[type="text"]'
    ).first

    pix_input.wait_for(
        state="visible",
        timeout=30000
    )

    # Dá um pequeno tempo para o checkout terminar de renderizar
    page.wait_for_timeout(1500)


    # ========================================================
    # 7. Obter PIX copia e cola
    # ========================================================

    pix_code = pix_input.input_value()

    print("\n==============================")
    print("PIX COPIA E COLA")
    print("==============================")
    print(pix_code)
    print("==============================")


    # ========================================================
    # 8. Procurar o QR Code
    # ========================================================

    print("\nProcurando QR Code...")

    qr_encontrado = False


    # Primeiro tenta canvas
    canvases = page.locator("canvas")

    for i in range(canvases.count()):

        canvas = canvases.nth(i)

        if not canvas.is_visible():
            continue

        box = canvas.bounding_box()

        if not box:
            continue

        largura = box["width"]
        altura = box["height"]

        print(
            f"Canvas {i}: "
            f"{largura:.0f}x{altura:.0f}"
        )

        # QR normalmente é quadrado
        if (
            largura >= 150
            and altura >= 150
            and abs(largura - altura) <= 50
        ):

            canvas.screenshot(
                path="qrcode.png"
            )

            print(
                "\nQR Code salvo em: qrcode.png"
            )

            qr_encontrado = True
            break


    # ========================================================
    # 9. Caso não encontre no canvas, tenta SVG
    # ========================================================

    if not qr_encontrado:

        svgs = page.locator("svg")

        for i in range(svgs.count()):

            svg = svgs.nth(i)

            if not svg.is_visible():
                continue

            box = svg.bounding_box()

            if not box:
                continue

            largura = box["width"]
            altura = box["height"]

            print(
                f"SVG {i}: "
                f"{largura:.0f}x{altura:.0f}"
            )

            if (
                largura >= 150
                and altura >= 150
                and abs(largura - altura) <= 50
            ):

                svg.screenshot(
                    path="qrcode.png"
                )

                print(
                    "\nQR Code salvo em: qrcode.png"
                )

                qr_encontrado = True
                break


    # ========================================================
    # 10. Se não encontrou, salva a página inteira
    # ========================================================

    if not qr_encontrado:

        print(
            "\nNão consegui identificar "
            "automaticamente o QR Code."
        )

        page.screenshot(
            path="checkout.png",
            full_page=True
        )

        print(
            "Screenshot salvo em: checkout.png"
        )


    # ========================================================
    # 11. Manter navegador aberto
    # ========================================================

    input(
        "\nPressione ENTER para fechar o navegador..."
    )

    browser.close()
