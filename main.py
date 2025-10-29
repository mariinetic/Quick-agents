from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()


def main():
    print("Hello from langchain-course!")
    information = """
 Ariana Grande-Butera[2][3] (Boca Raton, 26 de junho de 1993)[4][5] é uma cantora, compositora e atriz norte-americana. Ao longo da carreira, tornou-se uma das cantoras mais ouvidas da história da música em streaming[6] e um dos nomes de maior relevância da música pop na atualidade.[7] Grande é conhecida por seu alcance vocal de quatro oitavas e seu uso característico do registro de apito.[8] Ela começou sua carreira como atriz aparecendo no musical 13 (2008), da Broadway, e alcançou a fama como Cat Valentine na série de televisão Victorious (2010–2013), da Nickelodeon, e seu spin-off Sam & Cat (2013–2014). A artista assinou com a Republic Records e lançou seu álbum de estreia Yours Truly (2013), que apresenta uma mistura de pop e R&B. Já seus dois discos seguintes, My Everything (2014) e Dangerous Woman (2016) – que apresentam mais elementos eletrônicos –, solidificaram seu sucesso comercial e de crítica. Ambos os álbuns continham os singles de sucesso internacional; "Problem", "Break Free", "Bang Bang", "One Last Time", e "Side to Side"

Os conflitos pessoais da artista inspiraram o conteúdo lírico dos álbuns de trap, Sweetener (2018) e Thank U, Next (2019). O primeiro rendeu a Grande seu primeiro prêmio Grammy, enquanto o último rendeu os seus primeiros singles número um na tabela estadunidense Billboard Hot 100; "Thank U Next" e "7 Rings", o que a tornou a primeira artista solo a ocupar as três primeiras posições dessa parada musical. Posteriormente, ela alcançou o maior número de estreias número um na história da Hot 100 com a faixa-título de seu sexto álbum, Positions (2020), bem como as colaborações "Stuck with U", com Justin Bieber, e "Rain on Me", com Lady Gaga. Após 4 anos de seu último álbum, Grande lançou o aclamado pela crítica, Eternal Sunshine (2024), que se tornou seu sexto álbum no topo da tabela Billboard 200 e gerou dois singles número um "Yes, And?" e "We Can't Be Friends (Wait for Your Love)". Ela voltou a atuar no cinema com a sátira política Don't Look Up (2021) e recebeu elogios da crítica por sua interpretação de Glinda the Good, no musical de fantasia Wicked (2024), que lhe rendeu uma indicação ao Oscar de Melhor Atriz Coadjuvante.

Grande está entre os artistas musicais que mais venderam no mundo – com mais de 90 milhões de gravações comercializadas –, e é a quinta musicista com mais singles certificados por vendas digitais. Entre seus vários prêmios e reconhecimentos há dois Grammys, um Brit Award, três Billboard Music Awards e American Music Awards, dez MTV Video Music Awards, e 36 recordes mundiais no Guinness World Records. Ela foi nomeada duas vezes a Melhor Artista Feminina do Ano (2017 e 2019), a Mulher do Ano (2018) e a artista feminina de maior sucesso a surgir na década de 2010, além da Billboard reconhece-la como a nona maior estrela pop do período 2000-2024. É destaque em listas das maiores cantoras de todos os tempos feita pela Rolling Stone, Time 100 (2016 e 2019), Forbes Celebrity 100 (2019–2020) e foi classificada como a musicista feminina mais bem paga de 2020 pela Forbes. Fora da música e do cinema, Grande trabalhou com muitas organizações de caridade e defensores dos direitos dos animais, saúde mental, igualdade de gênero, raça e LGBT. Seus empreendimentos comerciais incluem a R.E.M. Beauty, uma marca de cosméticos lançada em 2021, e uma linha de fragrâncias que arrecadou mais de 1 bilhão de dólares em vendas no varejo global. Influente nas redes sociais, é a sexta pessoa mais seguida no Instagram.
    """

    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    # llm = ChatOllama(temperature=0, model="gemma3:270m")
    llm = ChatOpenAI(temperature=0, model="gpt-5")
    chain = summary_prompt_template | llm

    response = chain.invoke(input={"information": information})
    print(response.content)

if __name__ == "__main__":
    main()
